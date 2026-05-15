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
    """Match the kernel's fast_cvt_float_to_e4m3fn helper exactly."""
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


def swiglu_dynamic_quant_reference(hidden_states, smooth_scale, dst_dtype):
    hidden_size = hidden_states.shape[1] // 2
    x1 = hidden_states[:, :hidden_size].float()
    x2 = hidden_states[:, hidden_size:].float()
    swiglu = torch.nn.functional.silu(x1) * x2
    scaled = swiglu * smooth_scale.float().view(1, -1)
    quant_tokens, per_token_scale = quantize_reference(scaled, dst_dtype)
    return quant_tokens.contiguous(), per_token_scale.contiguous(), scaled.contiguous()


def run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype):
    module = _load_extension("swiglu_dynamic_quant_sycl")
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] // 2
    quant_tokens = torch.empty((num_tokens, hidden_size), dtype=dst_dtype, device=hidden_states.device)
    per_token_scale = torch.empty((num_tokens,), dtype=torch.float32, device=hidden_states.device)
    module.swiglu_dynamic_quant(
        hidden_states.contiguous(),
        smooth_scale.contiguous(),
        quant_tokens,
        per_token_scale,
    )
    torch.xpu.synchronize()
    return quant_tokens, per_token_scale


def create_test_inputs(num_tokens, hidden_size, src_dtype=torch.bfloat16, device="xpu", seed=42):
    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens, hidden_size * 2, dtype=src_dtype, device=device)
    smooth_scale = torch.rand(hidden_size, dtype=torch.float32, device=device) + 0.5
    return hidden_states, smooth_scale


def verify_quant_output(quant_tokens, per_token_scale, expected, dst_dtype):
    max_dtype_val = _QUANT_MAX[dst_dtype]
    dequantized = quant_tokens.float() * per_token_scale.view(-1, 1)
    max_per_token = expected.abs().amax(dim=-1, keepdim=True)
    if dst_dtype == torch.int8:
        tolerance = torch.clamp(2.0 * max_per_token / max_dtype_val, min=1e-4)
    else:
        tolerance = torch.clamp(0.15 * max_per_token, min=1e-4)
    diff = (dequantized - expected).abs()
    return bool((diff <= tolerance).all().item()), diff.max().item()


class TestSwigluDynamicQuant:
    @pytest.mark.parametrize("num_tokens", [1, 40, 80, 4096])
    @pytest.mark.parametrize("hidden_size", [256, 2048, 3072, 7168])
    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_sweep_correctness(self, num_tokens, hidden_size, src_dtype, dst_dtype):
        hidden_states, smooth_scale = create_test_inputs(
            num_tokens, hidden_size, src_dtype=src_dtype
        )
        ref_quant_tokens, ref_per_scale, expected = swiglu_dynamic_quant_reference(
            hidden_states, smooth_scale, dst_dtype
        )

        out_quant_tokens, out_per_scale = run_swiglu_dynamic_quant(
            hidden_states, smooth_scale, dst_dtype
        )

        assert out_quant_tokens.shape == ref_quant_tokens.shape
        assert out_per_scale.shape == ref_per_scale.shape
        torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-5, rtol=1e-5)

        if dst_dtype == torch.int8:
            diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
            assert diff.max().item() <= 1

        passed, max_diff = verify_quant_output(
            out_quant_tokens, out_per_scale, expected, dst_dtype
        )
        assert passed, f"Max dequant error {max_diff} exceeds tolerance"

    @pytest.mark.parametrize("num_tokens,hidden_size", [
        (1, 128),
        (1, 4096),
        (4, 512),
        (32, 4096),
        (128, 4096),
        (512, 2048),
        (1, 14336),
        (32, 14336),
        (8, 64),
        (4, 192),
    ])
    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_edge_shape_correctness(self, num_tokens, hidden_size, src_dtype, dst_dtype):
        hidden_states, smooth_scale = create_test_inputs(
            num_tokens, hidden_size, src_dtype=src_dtype
        )
        quant_tokens, per_token_scale = run_swiglu_dynamic_quant(
            hidden_states, smooth_scale, dst_dtype
        )
        _, _, expected = swiglu_dynamic_quant_reference(hidden_states, smooth_scale, dst_dtype)

        max_dtype_val = _QUANT_MAX[dst_dtype]
        assert quant_tokens.shape == (num_tokens, hidden_size)
        assert per_token_scale.shape == (num_tokens,)
        assert quant_tokens.dtype == dst_dtype
        assert per_token_scale.dtype == torch.float32
        assert quant_tokens.float().min() >= -max_dtype_val
        assert quant_tokens.float().max() <= max_dtype_val

        passed, max_diff = verify_quant_output(
            quant_tokens, per_token_scale, expected, dst_dtype
        )
        assert passed, f"Max dequant error {max_diff} exceeds tolerance"

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_scale_positive(self, src_dtype, dst_dtype):
        hidden_states, smooth_scale = create_test_inputs(32, 4096, src_dtype=src_dtype)
        _, per_token_scale = run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype)
        assert (per_token_scale >= 0).all()

    @pytest.mark.parametrize("dst_dtype", list(_QUANT_MAX.keys()))
    def test_zero_input(self, dst_dtype):
        num_tokens, hidden_size = 4, 512
        hidden_states = torch.zeros(num_tokens, hidden_size * 2, dtype=torch.bfloat16, device="xpu")
        smooth_scale = torch.ones(hidden_size, dtype=torch.float32, device="xpu")

        quant_tokens, per_token_scale = run_swiglu_dynamic_quant(
            hidden_states, smooth_scale, dst_dtype
        )

        assert (quant_tokens.float() == 0).all()
        assert torch.allclose(per_token_scale, torch.ones_like(per_token_scale))

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_uniform_input(self, src_dtype, dst_dtype):
        num_tokens, hidden_size = 8, 256
        hidden_states = torch.full((num_tokens, hidden_size * 2), 1.0, dtype=src_dtype, device="xpu")
        smooth_scale = torch.ones(hidden_size, dtype=torch.float32, device="xpu")

        _, per_token_scale = run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype)

        assert torch.allclose(
            per_token_scale,
            per_token_scale[0:1].expand_as(per_token_scale),
            atol=1e-6,
        )

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_large_values(self, src_dtype, dst_dtype):
        num_tokens, hidden_size = 4, 512
        max_dtype_val = _QUANT_MAX[dst_dtype]
        torch.manual_seed(123)
        hidden_states = torch.randn(num_tokens, hidden_size * 2, dtype=src_dtype, device="xpu") * 100.0
        smooth_scale = torch.ones(hidden_size, dtype=torch.float32, device="xpu") * 2.0

        quant_tokens, _ = run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype)

        assert quant_tokens.float().min() >= -max_dtype_val
        assert quant_tokens.float().max() <= max_dtype_val

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_smooth_scale_effect(self, src_dtype, dst_dtype):
        num_tokens, hidden_size = 4, 256
        hidden_states, _ = create_test_inputs(num_tokens, hidden_size, src_dtype=src_dtype)

        scale_ones = torch.ones(hidden_size, dtype=torch.float32, device="xpu")
        _, scale1 = run_swiglu_dynamic_quant(hidden_states, scale_ones, dst_dtype)

        scale_twos = torch.full((hidden_size,), 2.0, dtype=torch.float32, device="xpu")
        _, scale2 = run_swiglu_dynamic_quant(hidden_states, scale_twos, dst_dtype)

        ratio = scale2 / scale1
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), rtol=0.01)

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_deterministic(self, src_dtype, dst_dtype):
        hidden_states, smooth_scale = create_test_inputs(16, 2048, src_dtype=src_dtype, seed=99)

        quant1, scale1 = run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype)
        quant2, scale2 = run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype)

        assert torch.equal(quant1, quant2)
        assert torch.equal(scale1, scale2)

    @pytest.mark.parametrize("src_dtype,dst_dtype", DTYPE_COMBOS)
    def test_per_token_independence(self, src_dtype, dst_dtype):
        num_tokens, hidden_size = 8, 512
        hidden_states, smooth_scale = create_test_inputs(num_tokens, hidden_size, src_dtype=src_dtype)

        quant1, scale1 = run_swiglu_dynamic_quant(hidden_states, smooth_scale, dst_dtype)

        modified = hidden_states.clone()
        modified[3] = torch.randn(hidden_size * 2, dtype=src_dtype, device="xpu")

        quant2, scale2 = run_swiglu_dynamic_quant(modified, smooth_scale, dst_dtype)

        for idx in range(num_tokens):
            if idx == 3:
                continue
            assert torch.equal(quant1[idx], quant2[idx])
            assert torch.equal(scale1[idx:idx + 1], scale2[idx:idx + 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
