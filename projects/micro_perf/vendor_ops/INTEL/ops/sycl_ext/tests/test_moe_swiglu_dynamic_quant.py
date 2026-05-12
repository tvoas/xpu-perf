import os
import importlib.util
import pytest
import torch

torch.manual_seed(42)

# Dynamically load the localized SYCL extension
so_path = os.path.join(os.path.dirname(__file__), "..", "moe_swiglu_dynamic_quant_sycl.so")
if not os.path.exists(so_path):
    raise RuntimeError(f"Could not find {so_path}. Did you run build.sh?")
spec = importlib.util.spec_from_file_location("moe_swiglu_dynamic_quant_sycl", so_path)
sycl_ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sycl_ext)


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
    return fp8_bits.view(torch.float8_e4m3fn)


def quantize_reference(values, dst_dtype):
    quant_max = 448.0 if dst_dtype == torch.float8_e4m3fn else 127.0
    scale = values.abs().amax(dim=-1) / quant_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    normalized = values / scale.unsqueeze(-1)
    if dst_dtype == torch.int8:
        quantized = torch.round(normalized).to(torch.int8)
    else:
        quantized = emulate_kernel_fp8_e4m3fn(normalized)

    return quantized, scale


def baseline_moe_swiglu(scatter_tokens, smooth_scale, scatter_expert_ids, dst_dtype):
    """Pure PyTorch vectorized baseline for MoE SwiGLU + Dynamic Quantization."""
    hidden_size = scatter_tokens.shape[1] // 2

    token_scales = smooth_scale[scatter_expert_ids.long()].float()

    x1 = scatter_tokens[:, :hidden_size].float()
    x2 = scatter_tokens[:, hidden_size:].float()

    x1_silu = torch.nn.functional.silu(x1)
    swiglu = x1_silu * x2
    scaled = swiglu * token_scales

    unquantized_math = scaled

    quant_tokens, per_token_scale = quantize_reference(scaled, dst_dtype)

    return quant_tokens, per_token_scale, unquantized_math


@pytest.mark.parametrize("num_scattered", [1, 64, 1024, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 2048, 8192])
@pytest.mark.parametrize("num_experts", [1, 64, 256])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.float8_e4m3fn])
def test_moe_swiglu_dynamic_quant(num_scattered, hidden_size, num_experts, src_dtype, dst_dtype):
    device = "xpu"

    experts_token_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    experts_token_start = torch.zeros(num_experts, dtype=torch.int32, device=device)

    tokens_per_expert = num_scattered // num_experts
    experts_token_count[:] = tokens_per_expert
    experts_token_count[-1] = num_scattered - (tokens_per_expert * (num_experts - 1))

    if num_experts > 1:
        experts_token_start[1:] = torch.cumsum(experts_token_count[:-1], dim=0)
    max_token_num = int(experts_token_count.max().item())
    scatter_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device),
        experts_token_count.to(torch.int64)
    )

    scatter_tokens = torch.randn((num_scattered, hidden_size * 2), dtype=src_dtype, device=device)
    smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    out_quant_tokens = torch.zeros((num_scattered, hidden_size), dtype=dst_dtype, device=device)
    out_per_scale = torch.zeros(num_scattered, dtype=torch.float32, device=device)

    # 1. Base Accuracy Reference
    ref_quant_tokens, ref_per_scale, ref_unquantized_tokens = baseline_moe_swiglu(
        scatter_tokens, smooth_scale, scatter_expert_ids, dst_dtype
    )

    out_quant_tokens.zero_()
    out_per_scale.zero_()

    # 2. XPU Kernel Output via sycl_ext
    sycl_ext.moe_swiglu_dynamic_quant(
        scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
        scatter_expert_ids,
        out_quant_tokens, out_per_scale, num_experts, max_token_num
    )
    torch.xpu.synchronize()

    # 3. Accuracy Verifications
    assert out_quant_tokens.shape == ref_quant_tokens.shape, "Output quant shape mismatch!"
    assert out_per_scale.shape == ref_per_scale.shape, "Output scale shape mismatch!"

    torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-5, rtol=1e-5)

    if dst_dtype == torch.int8:
        diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
        assert diff.max().item() <= 1, "Quantized values diverge completely!"

    custom_dequantized = out_quant_tokens.float() * out_per_scale.unsqueeze(1)
    ref_dequantized = ref_quant_tokens.float() * ref_per_scale.unsqueeze(1)

    float_diff = (custom_dequantized - ref_dequantized).abs()

    if dst_dtype == torch.int8:
        allowed_error = torch.maximum(out_per_scale, ref_per_scale).unsqueeze(1) + 1e-6
    else:
        allowed_error = 32.0 * torch.maximum(out_per_scale, ref_per_scale).unsqueeze(1) + 1e-3

    assert torch.all(float_diff <= allowed_error), (
        f"Dequantized outputs diverge too broadly. Max error: {float_diff.max().item()} > "
        f"Allowed: {allowed_error.max().item()}"
    )
