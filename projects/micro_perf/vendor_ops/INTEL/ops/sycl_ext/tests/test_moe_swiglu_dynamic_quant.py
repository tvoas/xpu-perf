import importlib.util
import itertools
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="SYCL extension tests require an available XPU device",
)

torch.manual_seed(42)

_FLOAT8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_QUANT_MAX = {torch.int8: 127.0}
if _FLOAT8_DTYPE is not None:
    _QUANT_MAX[_FLOAT8_DTYPE] = 448.0

DTYPE_COMBOS = [
    (torch.bfloat16, torch.int8),
    (torch.float16, torch.int8),
]
if _FLOAT8_DTYPE is not None:
    DTYPE_COMBOS.extend([
        (torch.bfloat16, _FLOAT8_DTYPE),
        (torch.float16, _FLOAT8_DTYPE),
    ])

_MODULE_DIR = Path(__file__).resolve().parents[1]
_EXTENSION_CACHE = {}


def _load_extension(module_name):
    if module_name in _EXTENSION_CACHE:
        return _EXTENSION_CACHE[module_name]

    module_path = _MODULE_DIR / f"{module_name}.so"
    if not module_path.exists():
        pytest.skip(f"Missing built extension: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _EXTENSION_CACHE[module_name] = module
    return module


def emulate_kernel_fp8_e4m3fn(values):
    values = values.float().contiguous()
    bits = values.view(torch.int32)
    sign = (bits >> 24) & 0x80
    abs_bits = bits & 0x7FFFFFFF
    rounded = abs_bits + 0x00080000

    exp = (rounded >> 23) - 127 + 7
    mantissa = (rounded & 0x7FFFFF) >> 20

    fp8_bits = torch.zeros_like(bits, dtype=torch.uint8)
    is_normal = (exp > 0) & (exp < 16)
    is_overflow = exp >= 16
    is_underflow = exp <= 0

    fp8_bits = torch.where(is_normal, (sign | (exp << 3) | mantissa).to(torch.uint8), fp8_bits)
    fp8_bits = torch.where(is_overflow, (sign | 0x7E).to(torch.uint8), fp8_bits)
    fp8_bits = torch.where(is_underflow, sign.to(torch.uint8), fp8_bits)
    return fp8_bits.view(_FLOAT8_DTYPE)


def quantize_reference(values, dst_dtype):
    quant_max = _QUANT_MAX[dst_dtype]
    scale = values.abs().amax(dim=-1) / quant_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    normalized = values / scale.unsqueeze(-1)
    if dst_dtype == torch.int8:
        quantized = torch.round(normalized).to(torch.int8)
    else:
        quantized = emulate_kernel_fp8_e4m3fn(normalized)

    return quantized, scale


def moe_swiglu_dynamic_quant_reference(
    scatter_tokens,
    experts_smooth_scale,
    experts_token_count,
    experts_token_start,
    dst_dtype,
):
    dispatch_tokens = scatter_tokens.shape[0]
    hidden_size = scatter_tokens.shape[1] // 2
    gate = scatter_tokens[:, :hidden_size].float()
    up = scatter_tokens[:, hidden_size:].float()
    swiglu = torch.nn.functional.silu(gate) * up
    quant_tokens = torch.empty((dispatch_tokens, hidden_size), dtype=dst_dtype, device=scatter_tokens.device)
    per_token_scale = torch.empty((dispatch_tokens,), dtype=torch.float32, device=scatter_tokens.device)

    for expert_idx in range(experts_token_count.shape[0]):
        start = experts_token_start[expert_idx].item()
        count = experts_token_count[expert_idx].item()
        if count <= 0:
            continue
        end = start + count
        expected = swiglu[start:end] * experts_smooth_scale[expert_idx].view(1, -1)
        quant, scale = quantize_reference(expected, dst_dtype)
        quant_tokens[start:end] = quant
        per_token_scale[start:end] = scale

    return quant_tokens.contiguous(), per_token_scale.contiguous(), swiglu.contiguous()


def run_moe_swiglu_dynamic_quant(
    scatter_tokens,
    experts_smooth_scale,
    experts_token_count,
    experts_token_start,
    scatter_expert_ids,
    dst_dtype,
):
    module = _load_extension("moe_swiglu_dynamic_quant_sycl")
    dispatch_tokens = scatter_tokens.shape[0]
    hidden_size = scatter_tokens.shape[1] // 2
    num_experts = experts_smooth_scale.shape[0]
    max_token_num = int(experts_token_count.max().item()) if experts_token_count.numel() > 0 else 0

    quant_tokens = torch.empty((dispatch_tokens, hidden_size), dtype=dst_dtype, device=scatter_tokens.device)
    per_token_scale = torch.empty((dispatch_tokens,), dtype=torch.float32, device=scatter_tokens.device)

    module.moe_swiglu_dynamic_quant(
        scatter_tokens.contiguous(),
        experts_smooth_scale.contiguous(),
        experts_token_count.contiguous(),
        experts_token_start.contiguous(),
        scatter_expert_ids.contiguous(),
        quant_tokens,
        per_token_scale,
        num_experts,
        max_token_num,
    )
    torch.xpu.synchronize()
    return quant_tokens, per_token_scale


def create_moe_test_inputs(dispatch_tokens, hidden_size, num_experts, src_dtype=torch.bfloat16, device="xpu", seed=42):
    torch.manual_seed(seed)

    scatter_tokens = torch.randn(dispatch_tokens, hidden_size * 2, dtype=src_dtype, device=device)
    experts_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device) + 0.5

    base_count = dispatch_tokens // num_experts
    remainder = dispatch_tokens % num_experts
    token_counts = [base_count + (1 if idx < remainder else 0) for idx in range(num_experts)]

    experts_token_count = torch.tensor(token_counts, dtype=torch.int32, device=device)
    offsets = [0] + list(itertools.accumulate(token_counts[:-1]))
    experts_token_start = torch.tensor(offsets, dtype=torch.int32, device=device)
    scatter_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device),
        experts_token_count.to(torch.int64),
    )

    return (
        scatter_tokens,
        experts_smooth_scale,
        experts_token_count,
        experts_token_start,
        scatter_expert_ids,
    )


def verify_moe_quant_output(
    quant_tokens,
    per_token_scale,
    scatter_tokens,
    experts_smooth_scale,
    experts_token_count,
    experts_token_start,
    dst_dtype,
):
    hidden_size = scatter_tokens.shape[1] // 2
    max_dtype_val = _QUANT_MAX[dst_dtype]
    gate = scatter_tokens[:, :hidden_size].float()
    up = scatter_tokens[:, hidden_size:].float()
    swiglu = torch.nn.functional.silu(gate) * up

    all_passed = True
    max_error = 0.0
    for expert_idx in range(experts_token_count.shape[0]):
        start = experts_token_start[expert_idx].item()
        count = experts_token_count[expert_idx].item()
        if count == 0:
            continue
        end = start + count
        expected = swiglu[start:end] * experts_smooth_scale[expert_idx].view(1, -1)
        dequant = quant_tokens[start:end].float() * per_token_scale[start:end].view(-1, 1)
        max_per_token = expected.abs().amax(dim=-1, keepdim=True)
        if dst_dtype == torch.int8:
            tolerance = torch.clamp(2.0 * max_per_token / max_dtype_val, min=1e-4)
        else:
            tolerance = torch.clamp(0.15 * max_per_token, min=1e-4)

        diff = (dequant - expected).abs()
        if not (diff <= tolerance).all():
            all_passed = False
        max_error = max(max_error, diff.max().item())

    return all_passed, max_error


class TestMoeSwigluDynamicQuant:
    @pytest.mark.parametrize("num_scattered", [1, 64, 1024, 4096])
    @pytest.mark.parametrize("hidden_size", [64, 128, 2048, 8192])
    @pytest.mark.parametrize("num_experts", [1, 64, 256])
    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_sweep_correctness(self, num_scattered, hidden_size, num_experts, src_dtype, dst_dtype):
        inputs = create_moe_test_inputs(
            num_scattered, hidden_size, num_experts, src_dtype=src_dtype
        )
        scatter_tokens, experts_smooth_scale, experts_token_count, experts_token_start, scatter_expert_ids = inputs

        ref_quant_tokens, ref_per_scale, _ = moe_swiglu_dynamic_quant_reference(
            scatter_tokens,
            experts_smooth_scale,
            experts_token_count,
            experts_token_start,
            dst_dtype,
        )
        out_quant_tokens, out_per_scale = run_moe_swiglu_dynamic_quant(
            scatter_tokens,
            experts_smooth_scale,
            experts_token_count,
            experts_token_start,
            scatter_expert_ids,
            dst_dtype,
        )

        assert out_quant_tokens.shape == ref_quant_tokens.shape
        assert out_per_scale.shape == ref_per_scale.shape
        torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-5, rtol=1e-5)

        if dst_dtype == torch.int8:
            diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
            assert diff.max().item() <= 1

        passed, max_err = verify_moe_quant_output(
            out_quant_tokens,
            out_per_scale,
            scatter_tokens,
            experts_smooth_scale,
            experts_token_count,
            experts_token_start,
            dst_dtype,
        )
        assert passed, f"Max dequant error {max_err} exceeds tolerance"

    @pytest.mark.parametrize("dispatch_tokens,hidden_size,num_experts", [
        (8, 256, 2),
        (16, 512, 4),
        (64, 4096, 8),
        (128, 4096, 8),
        (256, 4096, 16),
        (1024, 2048, 64),
        (80, 4096, 8),
        (33, 4096, 8),
        (12, 64, 4),
        (7, 192, 3),
    ])
    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_edge_shape_correctness(self, dispatch_tokens, hidden_size, num_experts, src_dtype, dst_dtype):
        inputs = create_moe_test_inputs(
            dispatch_tokens, hidden_size, num_experts, src_dtype=src_dtype
        )
        scatter_tokens, experts_smooth_scale, experts_token_count, experts_token_start, scatter_expert_ids = inputs

        quant_tokens, per_token_scale = run_moe_swiglu_dynamic_quant(
            scatter_tokens,
            experts_smooth_scale,
            experts_token_count,
            experts_token_start,
            scatter_expert_ids,
            dst_dtype,
        )

        max_dtype_val = _QUANT_MAX[dst_dtype]
        assert quant_tokens.shape == (dispatch_tokens, hidden_size)
        assert per_token_scale.shape == (dispatch_tokens,)
        assert quant_tokens.dtype == dst_dtype
        assert per_token_scale.dtype == torch.float32
        assert quant_tokens.float().min() >= -max_dtype_val
        assert quant_tokens.float().max() <= max_dtype_val

        passed, max_err = verify_moe_quant_output(
            quant_tokens,
            per_token_scale,
            scatter_tokens,
            experts_smooth_scale,
            experts_token_count,
            experts_token_start,
            dst_dtype,
        )
        assert passed, f"Max dequant error {max_err} exceeds tolerance"

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_single_expert(self, src_dtype, dst_dtype):
        inputs = create_moe_test_inputs(16, 512, 1, src_dtype=src_dtype)
        quant_tokens, per_token_scale = run_moe_swiglu_dynamic_quant(*inputs, dst_dtype)
        assert quant_tokens.shape == (16, 512)
        assert (per_token_scale >= 0).all()

    def test_expert_with_zero_tokens(self):
        hidden_size = 512
        num_experts = 4
        experts_token_count = torch.tensor([5, 0, 7, 0], dtype=torch.int32, device="xpu")
        experts_token_start = torch.tensor([0, 5, 5, 12], dtype=torch.int32, device="xpu")
        dispatch_tokens = 12
        scatter_expert_ids = torch.tensor([0] * 5 + [2] * 7, dtype=torch.int32, device="xpu")

        for src_dtype, dst_dtype in DTYPE_COMBOS:
            torch.manual_seed(42)
            scatter_tokens = torch.randn(dispatch_tokens, hidden_size * 2, dtype=src_dtype, device="xpu")
            experts_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device="xpu") + 0.5

            quant_tokens, _ = run_moe_swiglu_dynamic_quant(
                scatter_tokens,
                experts_smooth_scale,
                experts_token_count,
                experts_token_start,
                scatter_expert_ids,
                dst_dtype,
            )

            assert quant_tokens.shape == (dispatch_tokens, hidden_size)
            assert quant_tokens.dtype == dst_dtype

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_per_expert_scale_isolation(self, src_dtype, dst_dtype):
        dispatch_tokens = 16
        hidden_size = 256
        num_experts = 4
        tokens_per_expert = dispatch_tokens // num_experts

        torch.manual_seed(42)
        scatter_tokens = torch.randn(dispatch_tokens, hidden_size * 2, dtype=src_dtype, device="xpu")
        experts_smooth_scale = torch.ones(num_experts, hidden_size, dtype=torch.float32, device="xpu")
        experts_token_count = torch.full((num_experts,), tokens_per_expert, dtype=torch.int32, device="xpu")
        experts_token_start = torch.tensor(
            [idx * tokens_per_expert for idx in range(num_experts)],
            dtype=torch.int32,
            device="xpu",
        )
        scatter_expert_ids = torch.repeat_interleave(
            torch.arange(num_experts, dtype=torch.int32, device="xpu"),
            experts_token_count.to(torch.int64),
        )

        quant1, scale1 = run_moe_swiglu_dynamic_quant(
            scatter_tokens,
            experts_smooth_scale,
            experts_token_count,
            experts_token_start,
            scatter_expert_ids,
            dst_dtype,
        )

        modified_scale = experts_smooth_scale.clone()
        modified_scale[2] *= 2.0
        quant2, scale2 = run_moe_swiglu_dynamic_quant(
            scatter_tokens,
            modified_scale,
            experts_token_count,
            experts_token_start,
            scatter_expert_ids,
            dst_dtype,
        )

        for expert_idx in [0, 1, 3]:
            offset = experts_token_start[expert_idx].item()
            count = experts_token_count[expert_idx].item()
            assert torch.equal(quant1[offset:offset + count], quant2[offset:offset + count])
            assert torch.equal(scale1[offset:offset + count], scale2[offset:offset + count])

        offset = experts_token_start[2].item()
        count = experts_token_count[2].item()
        ratio = scale2[offset:offset + count] / scale1[offset:offset + count]
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), rtol=0.01)

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_deterministic(self, src_dtype, dst_dtype):
        inputs = create_moe_test_inputs(64, 2048, 8, src_dtype=src_dtype, seed=77)

        quant1, scale1 = run_moe_swiglu_dynamic_quant(*inputs, dst_dtype)
        quant2, scale2 = run_moe_swiglu_dynamic_quant(*inputs, dst_dtype)

        assert torch.equal(quant1, quant2)
        assert torch.equal(scale1, scale2)

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_scales_non_negative(self, src_dtype, dst_dtype):
        inputs = create_moe_test_inputs(128, 4096, 8, src_dtype=src_dtype)
        _, per_token_scale = run_moe_swiglu_dynamic_quant(*inputs, dst_dtype)
        assert (per_token_scale >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
