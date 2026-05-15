import importlib.util
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


def moe_scatter_dynamic_quant_reference(
    selected_experts,
    moe_weights,
    hidden_states,
    experts_smooth_scale,
    total_experts,
    dst_dtype,
    kernel_token_to_scatter_offset=None,
):
    num_tokens, topk = selected_experts.shape
    hidden_size = hidden_states.shape[1]
    device = hidden_states.device

    experts_token_count = torch.zeros(total_experts, dtype=torch.int32, device=device)

    if kernel_token_to_scatter_offset is None:
        token_to_scatter_offset = torch.zeros_like(selected_experts, dtype=torch.int32)
        for token_idx in range(num_tokens):
            for k_idx in range(topk):
                expert_idx = int(selected_experts[token_idx, k_idx].item())
                token_to_scatter_offset[token_idx, k_idx] = experts_token_count[expert_idx]
                experts_token_count[expert_idx] += 1
    else:
        token_to_scatter_offset = kernel_token_to_scatter_offset
        for token_idx in range(num_tokens):
            for k_idx in range(topk):
                expert_idx = int(selected_experts[token_idx, k_idx].item())
                experts_token_count[expert_idx] += 1

    experts_token_start = torch.zeros_like(experts_token_count)
    if total_experts > 1:
        experts_token_start[1:] = torch.cumsum(experts_token_count[:-1], dim=0)

    flat_experts = selected_experts.reshape(-1).long()
    flat_offsets = token_to_scatter_offset.reshape(-1)
    target_idx = experts_token_start[flat_experts] + flat_offsets

    repeated_hidden = hidden_states.float().repeat_interleave(topk, dim=0)
    repeated_weights = moe_weights.float().reshape(-1, 1)
    repeated_scales = experts_smooth_scale.float()[flat_experts]
    expected_rows = repeated_hidden * repeated_scales * repeated_weights
    quantized, per_token_scale = quantize_reference(expected_rows, dst_dtype)

    dispatch_tokens = num_tokens * topk
    scatter_tokens = torch.zeros((dispatch_tokens, hidden_size), dtype=dst_dtype, device=device)
    scatter_tokens[target_idx] = quantized

    scatter_per_token_scale = torch.zeros((dispatch_tokens,), dtype=torch.float32, device=device)
    scatter_per_token_scale[target_idx] = per_token_scale

    scatter_tokens_offset = torch.zeros((dispatch_tokens,), dtype=torch.int32, device=device)
    flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(topk).to(torch.int32)
    scatter_tokens_offset[target_idx] = flat_token_ids

    expected = torch.zeros((dispatch_tokens, hidden_size), dtype=torch.float32, device=device)
    expected[target_idx] = expected_rows

    return (
        token_to_scatter_offset,
        experts_token_count,
        experts_token_start,
        scatter_tokens,
        scatter_per_token_scale,
        scatter_tokens_offset,
        expected,
    )


def run_moe_scatter_dynamic_quant(
    hidden_states,
    experts_smooth_scale,
    selected_experts,
    moe_weights,
    dst_dtype,
    shared_experts_num=0,
):
    module = _load_extension("moe_scatter_dynamic_quant_sycl")
    num_tokens, total_topk = selected_experts.shape
    dispatch_tokens = num_tokens * total_topk
    hidden_size = hidden_states.shape[1]
    total_experts = experts_smooth_scale.shape[0]
    device = hidden_states.device

    token_to_scatter_offset = torch.empty_like(selected_experts, dtype=torch.int32)
    experts_token_count = torch.empty((total_experts,), dtype=torch.int32, device=device)
    experts_token_start = torch.empty((total_experts,), dtype=torch.int32, device=device)
    scatter_tokens = torch.empty((dispatch_tokens, hidden_size), dtype=dst_dtype, device=device)
    scatter_per_token_scale = torch.empty((dispatch_tokens,), dtype=torch.float32, device=device)
    scatter_tokens_offset = torch.empty((dispatch_tokens,), dtype=torch.int32, device=device)

    module.moe_scatter_dynamic_quant(
        selected_experts.contiguous(),
        moe_weights.contiguous(),
        token_to_scatter_offset,
        experts_token_count,
        experts_token_start,
        hidden_states.contiguous(),
        experts_smooth_scale.contiguous(),
        scatter_tokens,
        scatter_per_token_scale,
        scatter_tokens_offset,
        int(shared_experts_num),
    )
    torch.xpu.synchronize()
    return (
        token_to_scatter_offset,
        experts_token_count,
        experts_token_start,
        scatter_tokens,
        scatter_per_token_scale,
        scatter_tokens_offset,
    )


def create_scatter_test_inputs(
    num_tokens,
    hidden_size,
    num_experts,
    tokens_per_expert=None,
    src_dtype=torch.bfloat16,
    device="xpu",
    seed=42,
):
    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=src_dtype, device=device)
    experts_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device) + 0.5

    if tokens_per_expert is None:
        token_counts = [num_tokens // num_experts] * num_experts
        for idx in range(num_tokens % num_experts):
            token_counts[idx] += 1
    else:
        token_counts = list(tokens_per_expert)
        assert sum(token_counts) == num_tokens

    assignments = []
    for expert_idx, count in enumerate(token_counts):
        assignments.extend([expert_idx] * count)

    selected_experts = torch.tensor(assignments, dtype=torch.int32, device=device).view(num_tokens, 1)
    moe_weights = torch.ones((num_tokens, 1), dtype=torch.float32, device=device)
    return hidden_states, experts_smooth_scale, selected_experts, moe_weights


def build_dispatch_expert_ids(experts_token_count, experts_token_start):
    dispatch_tokens = int(experts_token_count.sum().item())
    device = experts_token_count.device
    expert_ids = torch.zeros(dispatch_tokens, dtype=torch.int64, device=device)
    for expert_idx in range(experts_token_count.numel()):
        start = experts_token_start[expert_idx].item()
        count = experts_token_count[expert_idx].item()
        if count > 0:
            expert_ids[start:start + count] = expert_idx
    return expert_ids


def canonicalize_scatter_outputs(
    scatter_tokens,
    scatter_per_token_scale,
    scatter_tokens_offset,
    experts_token_count,
    experts_token_start,
    num_tokens,
):
    expert_ids = build_dispatch_expert_ids(experts_token_count, experts_token_start)
    sort_key = (expert_ids * num_tokens) + scatter_tokens_offset.to(torch.int64)
    sort_idx = torch.argsort(sort_key)
    return (
        scatter_tokens[sort_idx],
        scatter_per_token_scale[sort_idx],
        scatter_tokens_offset[sort_idx],
        sort_idx,
    )


def verify_scatter_quant_output(scatter_tokens, scatter_per_token_scale, expected, dst_dtype):
    max_dtype_val = _QUANT_MAX[dst_dtype]
    dequant = scatter_tokens.float() * scatter_per_token_scale.view(-1, 1)
    max_per_token = expected.abs().amax(dim=-1, keepdim=True)
    if dst_dtype == torch.int8:
        tolerance = torch.clamp(2.0 * max_per_token / max_dtype_val, min=1e-4)
    else:
        tolerance = torch.clamp(0.15 * max_per_token, min=1e-4)
    diff = (dequant - expected).abs()
    return bool((diff <= tolerance).all().item()), diff.max().item()


class TestMoeScatterDynamicQuant:
    @pytest.mark.parametrize("num_tokens", [1, 64, 1024])
    @pytest.mark.parametrize("hidden_size", [64, 128, 2048])
    @pytest.mark.parametrize("topk", [5, 8])
    @pytest.mark.parametrize("num_experts", [8, 256])
    @pytest.mark.parametrize("shared_experts_num", [0, 2])
    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_sweep_correctness(self, num_tokens, hidden_size, topk, num_experts, shared_experts_num, src_dtype, dst_dtype):
        device = "xpu"
        total_experts = num_experts + shared_experts_num
        total_topk = topk + shared_experts_num

        routed_experts = torch.rand((num_tokens, num_experts), device=device).topk(topk, dim=-1).indices.to(torch.int32)
        if shared_experts_num > 0:
            shared_ids = torch.arange(num_experts, total_experts, dtype=torch.int32, device=device).unsqueeze(0).expand(num_tokens, shared_experts_num)
            selected_experts = torch.cat([routed_experts, shared_ids], dim=-1)
        else:
            selected_experts = routed_experts

        moe_weights = torch.rand((num_tokens, total_topk), dtype=torch.float32, device=device)
        hidden_states = torch.randn((num_tokens, hidden_size), dtype=src_dtype, device=device)
        experts_smooth_scale = torch.rand((total_experts, hidden_size), dtype=torch.float32, device=device)

        out_t_offset, out_t_count, out_t_start, out_scatter_tokens, out_per_scale, out_tokens_offset = run_moe_scatter_dynamic_quant(
            hidden_states,
            experts_smooth_scale,
            selected_experts,
            moe_weights,
            dst_dtype,
            shared_experts_num,
        )

        ref_t_offset, ref_t_count, ref_t_start, ref_scatter_tokens, ref_per_scale, ref_tokens_offset, expected = moe_scatter_dynamic_quant_reference(
            selected_experts,
            moe_weights,
            hidden_states,
            experts_smooth_scale,
            total_experts,
            dst_dtype,
            kernel_token_to_scatter_offset=out_t_offset,
        )

        torch.testing.assert_close(out_t_count, ref_t_count)
        torch.testing.assert_close(out_t_start, ref_t_start)

        sorted_custom_tokens, sorted_custom_scale, sorted_custom_offsets, _ = canonicalize_scatter_outputs(
            out_scatter_tokens,
            out_per_scale,
            out_tokens_offset,
            out_t_count,
            out_t_start,
            num_tokens,
        )
        sorted_ref_tokens, sorted_ref_scale, sorted_ref_offsets, sorted_ref_idx = canonicalize_scatter_outputs(
            ref_scatter_tokens,
            ref_per_scale,
            ref_tokens_offset,
            ref_t_count,
            ref_t_start,
            num_tokens,
        )

        torch.testing.assert_close(sorted_custom_offsets, sorted_ref_offsets)
        torch.testing.assert_close(sorted_custom_scale, sorted_ref_scale, atol=1e-5, rtol=1e-5)

        if dst_dtype == torch.int8:
            diff = (sorted_custom_tokens.int() - sorted_ref_tokens.int()).abs()
            assert diff.max().item() <= 1

        passed, max_err = verify_scatter_quant_output(
            sorted_custom_tokens,
            sorted_custom_scale,
            expected[sorted_ref_idx],
            dst_dtype,
        )
        assert passed, f"Max dequant error {max_err} exceeds tolerance"

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_single_expert(self, src_dtype, dst_dtype):
        inputs = create_scatter_test_inputs(32, 512, 1, src_dtype=src_dtype)
        _, _, _, scatter_tokens, scale, _ = run_moe_scatter_dynamic_quant(*inputs, dst_dtype)
        assert scatter_tokens.shape[0] == 32
        assert (scale >= 0).all()

    def test_expert_with_zero_tokens(self):
        num_tokens = 12
        hidden_size = 512
        num_experts = 4

        for src_dtype, dst_dtype in DTYPE_COMBOS:
            torch.manual_seed(42)
            hidden_states = torch.randn(num_tokens, hidden_size, dtype=src_dtype, device="xpu")
            experts_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device="xpu") + 0.5
            selected_experts = torch.tensor([[0], [0], [0], [0], [0], [2], [2], [2], [2], [2], [2], [2]], dtype=torch.int32, device="xpu")
            moe_weights = torch.ones((num_tokens, 1), dtype=torch.float32, device="xpu")

            _, experts_token_count, _, scatter_tokens, _, _ = run_moe_scatter_dynamic_quant(
                hidden_states,
                experts_smooth_scale,
                selected_experts,
                moe_weights,
                dst_dtype,
            )

            assert scatter_tokens.shape == (num_tokens, hidden_size)
            assert scatter_tokens.dtype == dst_dtype
            assert experts_token_count.tolist() == [5, 0, 7, 0]

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_duplicate_token_ids(self, src_dtype, dst_dtype):
        num_tokens = 16
        hidden_size = 256
        num_experts = 4

        torch.manual_seed(42)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=src_dtype, device="xpu")
        experts_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device="xpu") + 0.5
        selected_experts = torch.arange(num_experts, dtype=torch.int32, device="xpu").repeat(num_tokens, 1)
        moe_weights = torch.ones((num_tokens, num_experts), dtype=torch.float32, device="xpu")

        out_t_offset, out_t_count, out_t_start, scatter_tokens, scale, scatter_offsets = run_moe_scatter_dynamic_quant(
            hidden_states,
            experts_smooth_scale,
            selected_experts,
            moe_weights,
            dst_dtype,
        )
        *_, expected = moe_scatter_dynamic_quant_reference(
            selected_experts,
            moe_weights,
            hidden_states,
            experts_smooth_scale,
            num_experts,
            dst_dtype,
            kernel_token_to_scatter_offset=out_t_offset,
        )

        sorted_tokens, sorted_scale, _, sorted_idx = canonicalize_scatter_outputs(
            scatter_tokens,
            scale,
            scatter_offsets,
            out_t_count,
            out_t_start,
            num_tokens,
        )

        assert scatter_tokens.shape == (num_tokens * num_experts, hidden_size)
        passed, _ = verify_scatter_quant_output(sorted_tokens, sorted_scale, expected[sorted_idx], dst_dtype)
        assert passed

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_scatter_correctness(self, src_dtype, dst_dtype):
        num_tokens = 6
        hidden_size = 128
        num_experts = 2

        hidden_states = torch.randn(num_tokens, hidden_size, dtype=src_dtype, device="xpu")
        experts_smooth_scale = torch.ones(num_experts, hidden_size, dtype=torch.float32, device="xpu")
        selected_experts = torch.tensor([[0], [1], [0], [1], [0], [1]], dtype=torch.int32, device="xpu")
        moe_weights = torch.ones((num_tokens, 1), dtype=torch.float32, device="xpu")

        _, experts_token_count, experts_token_start, _, _, scatter_offsets = run_moe_scatter_dynamic_quant(
            hidden_states,
            experts_smooth_scale,
            selected_experts,
            moe_weights,
            dst_dtype,
        )

        assert experts_token_count.tolist() == [3, 3]
        assert experts_token_start.tolist() == [0, 3]
        expert0 = torch.sort(scatter_offsets[:3]).values.tolist()
        expert1 = torch.sort(scatter_offsets[3:]).values.tolist()
        assert expert0 == [0, 2, 4]
        assert expert1 == [1, 3, 5]

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_per_expert_scale_isolation(self, src_dtype, dst_dtype):
        num_tokens = 32
        hidden_size = 256
        num_experts = 4
        tokens_per_expert = num_tokens // num_experts

        inputs = create_scatter_test_inputs(
            num_tokens,
            hidden_size,
            num_experts,
            tokens_per_expert=[tokens_per_expert] * num_experts,
            src_dtype=src_dtype,
        )
        hidden_states, experts_smooth_scale, selected_experts, moe_weights = inputs

        _, experts_token_count, experts_token_start, scatter1, scale1, offsets1 = run_moe_scatter_dynamic_quant(
            hidden_states,
            experts_smooth_scale,
            selected_experts,
            moe_weights,
            dst_dtype,
        )

        modified_scale = experts_smooth_scale.clone()
        modified_scale[2] *= 3.0
        _, experts_token_count2, experts_token_start2, scatter2, scale2, offsets2 = run_moe_scatter_dynamic_quant(
            hidden_states,
            modified_scale,
            selected_experts,
            moe_weights,
            dst_dtype,
        )

        sorted_scatter1, sorted_scale1, sorted_offsets1, _ = canonicalize_scatter_outputs(
            scatter1,
            scale1,
            offsets1,
            experts_token_count,
            experts_token_start,
            num_tokens,
        )
        sorted_scatter2, sorted_scale2, sorted_offsets2, _ = canonicalize_scatter_outputs(
            scatter2,
            scale2,
            offsets2,
            experts_token_count2,
            experts_token_start2,
            num_tokens,
        )

        for expert_idx in [0, 1, 3]:
            offset = experts_token_start[expert_idx].item()
            count = experts_token_count[expert_idx].item()
            assert torch.equal(sorted_offsets1[offset:offset + count], sorted_offsets2[offset:offset + count])
            assert torch.equal(sorted_scatter1[offset:offset + count], sorted_scatter2[offset:offset + count])
            assert torch.equal(sorted_scale1[offset:offset + count], sorted_scale2[offset:offset + count])

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_deterministic(self, src_dtype, dst_dtype):
        inputs = create_scatter_test_inputs(64, 2048, 8, src_dtype=src_dtype, seed=77)
        scatter1 = run_moe_scatter_dynamic_quant(*inputs, dst_dtype)
        scatter2 = run_moe_scatter_dynamic_quant(*inputs, dst_dtype)

        sorted1 = canonicalize_scatter_outputs(
            scatter1[3],
            scatter1[4],
            scatter1[5],
            scatter1[1],
            scatter1[2],
            64,
        )
        sorted2 = canonicalize_scatter_outputs(
            scatter2[3],
            scatter2[4],
            scatter2[5],
            scatter2[1],
            scatter2[2],
            64,
        )

        assert torch.equal(sorted1[0], sorted2[0])
        assert torch.equal(sorted1[1], sorted2[1])
        assert torch.equal(sorted1[2], sorted2[2])

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_scales_non_negative(self, src_dtype, dst_dtype):
        inputs = create_scatter_test_inputs(128, 4096, 8, src_dtype=src_dtype)
        _, _, _, _, scale, _ = run_moe_scatter_dynamic_quant(*inputs, dst_dtype)
        assert (scale >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
