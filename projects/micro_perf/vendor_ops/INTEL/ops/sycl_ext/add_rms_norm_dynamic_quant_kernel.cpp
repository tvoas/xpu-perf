// Fused add-residual + RMS-norm + smooth-scale dynamic int8 quantization.
//
// Per token (row m, hidden dim k):
//   after_res[m,k]       = hidden_states[m,k] + residual[m,k]   (if has_residual)
//                        = hidden_states[m,k]                    (otherwise)
//   rms                  = sqrt( mean_k( after_res[m,k]^2 ) + eps )
//   after_norm[m,k]      = after_res[m,k] / rms * norm_weight[k]
//   scaled               = after_norm[m,k] * smooth_scale[k]
//   amax[m]              = max_k |scaled|
//   per_token_scale[m]   = amax[m] / 127
//   quant_tokens[m,k]    = round_to_nearest_even(scaled / per_token_scale[m])

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <cstdint>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

namespace {

void add_rms_norm_dynamic_quant_launch(
    fp16* input_ptr,
    fp16* residual_ptr,
    float* weight_ptr,
    float* smooth_scale_ptr,
    fp16* after_res_ptr,
    fp16* after_norm_ptr,
    int8_t* quant_tokens_ptr,
    float* per_token_scale_ptr,
    int const num_tokens,
    int const hidden_size,
    bool const has_residual,
    float const eps,
    sycl::queue& queue)
{
    constexpr int block_size = 256;
    int const blocks = hidden_size / block_size;
    constexpr int max_block = 32;  // 8192 / 256
    int const quant_offset = max_block * static_cast<int>(sizeof(float));

    sycl::range<2> GlobalRange(num_tokens, blocks);
    sycl::range<2> LocalRange(1, blocks);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
            slm_init(max_block * 2 * sizeof(float));

            int const token_idx = item.get_global_id(0);
            int const block_idx = item.get_local_id(1);

            fp16*  input_base      = input_ptr     + token_idx * hidden_size + block_idx * block_size;
            fp16*  after_res_base  = after_res_ptr  + token_idx * hidden_size + block_idx * block_size;
            fp16*  after_norm_base = after_norm_ptr + token_idx * hidden_size + block_idx * block_size;
            float* weight_base    = weight_ptr     + block_idx * block_size;

            // ---- residual stage ----
            simd<fp16, block_size> input = block_load<fp16, block_size>(input_base);
            if (has_residual) {
                fp16* residual_base = residual_ptr + token_idx * hidden_size + block_idx * block_size;
                simd<fp16, block_size> res = block_load<fp16, block_size>(residual_base);
                input += res;
            }
            block_store<fp16, block_size>(after_res_base, input);

            // ---- rms norm stage ----
            simd<float, block_size> xv_f32 = input;
            simd<float, block_size> accv = xv_f32 * xv_f32;
            float acc = sycl::ext::intel::esimd::detail::sum<float, float, block_size>(accv) / hidden_size;

            slm_block_store<float, 1>(block_idx * sizeof(float), acc);

            barrier();

            simd<float, max_block> slm_sum = slm_block_load<float, max_block>(0);
            float all_sum = sycl::ext::intel::esimd::detail::sum<float, float, max_block>(slm_sum);
            simd<float, 1> tmp_v = all_sum + eps;
            tmp_v = sycl::ext::intel::esimd::rsqrt(tmp_v);
            float scale = tmp_v[0];

            // norm_weight is float32 in our interface (fp16 in original IPEX)
            simd<float, block_size> yv_f32 = block_load<float, block_size>(weight_base);
            simd<fp16, block_size> result = xv_f32 * scale * yv_f32;
            block_store<fp16, block_size>(after_norm_base, result);

            // ---- quant stage ----
            int8_t* quant_tokens_base = quant_tokens_ptr + token_idx * hidden_size + block_idx * block_size;
            float*  per_token_scale_base = per_token_scale_ptr + token_idx;
            float*  smooth_scale_base = smooth_scale_ptr + block_idx * block_size;

            simd<float, block_size> smooth_scale = block_load<float, block_size>(smooth_scale_base);
            simd<float, block_size> smooth_input = smooth_scale * result;
            simd<float, block_size> smooth_input_abs = sycl::ext::intel::esimd::abs(smooth_input);
            float maxvalue = hmax<float, float, block_size>(smooth_input_abs);
            slm_block_store<float, 1>(quant_offset + block_idx * sizeof(float), maxvalue);

            barrier();

            simd<float, max_block> all_maxvalue = slm_block_load<float, max_block>(quant_offset);
            float global_maxvalue = hmax<float, float, max_block>(all_maxvalue) / 127.0f;
            if (block_idx == 0) {
                block_store<float, 1>(per_token_scale_base, global_maxvalue);
            }
            simd<float, block_size> quant = rnde<float>(smooth_input / global_maxvalue);
            simd<int8_t, block_size> quant_tokens = quant;
            block_store<int8_t, block_size>(quant_tokens_base, quant_tokens);
        });
    });
}

// ---------- host launcher ----------
void add_rms_norm_dynamic_quant_forward(
    at::Tensor const& hidden_states,
    at::Tensor const& residual,
    at::Tensor const& norm_weight,
    at::Tensor const& smooth_scale,
    at::Tensor& quant_out,
    at::Tensor& per_token_scale,
    at::Tensor& after_res,
    at::Tensor& after_norm,
    double eps) {
  TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
  TORCH_CHECK(quant_out.is_contiguous(), "quant_out must be contiguous");
  TORCH_CHECK(after_res.is_contiguous(), "after_res must be contiguous");
  TORCH_CHECK(after_norm.is_contiguous(), "after_norm must be contiguous");
  TORCH_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");
  TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2-D");
  TORCH_CHECK(quant_out.scalar_type() == at::kChar, "quant_out must be int8");
  TORCH_CHECK(
      norm_weight.scalar_type() == at::kFloat, "norm_weight must be fp32");
  TORCH_CHECK(
      smooth_scale.scalar_type() == at::kFloat, "smooth_scale must be fp32");
  TORCH_CHECK(
      per_token_scale.scalar_type() == at::kFloat,
      "per_token_scale must be fp32");
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kHalf,
      "kernel requires float16 hidden_states");

  int const hidden_size = hidden_states.size(1);
  int const num_tokens = hidden_states.size(0);

  TORCH_CHECK(
      hidden_size % 256 == 0, "hidden_size must be a multiple of 256");
  TORCH_CHECK(hidden_size <= 8192, "hidden_size must be <= 8192");

  bool const has_residual = residual.numel() > 0;
  if (has_residual) {
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(
        residual.sizes() == hidden_states.sizes(),
        "residual shape must match hidden_states");
  }

  TORCH_CHECK(
      norm_weight.numel() == hidden_size,
      "norm_weight numel must equal hidden_size");
  TORCH_CHECK(
      smooth_scale.numel() == hidden_size,
      "smooth_scale numel must equal hidden_size");
  TORCH_CHECK(
      per_token_scale.numel() == num_tokens,
      "per_token_scale numel must equal num_tokens");

  if (num_tokens == 0) return;

  auto& queue =
      c10::xpu::getCurrentXPUStream(hidden_states.device().index()).queue();

  add_rms_norm_dynamic_quant_launch(
      reinterpret_cast<fp16*>(hidden_states.data_ptr()),
      has_residual ? reinterpret_cast<fp16*>(residual.data_ptr()) : nullptr,
      norm_weight.data_ptr<float>(),
      smooth_scale.data_ptr<float>(),
      reinterpret_cast<fp16*>(after_res.data_ptr()),
      reinterpret_cast<fp16*>(after_norm.data_ptr()),
      quant_out.data_ptr<int8_t>(),
      per_token_scale.data_ptr<float>(),
      num_tokens, hidden_size, has_residual,
      static_cast<float>(eps),
      queue);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "add_rms_norm_dynamic_quant_forward",
      &add_rms_norm_dynamic_quant_forward,
      "Fused add-residual + RMS-norm + smooth-scale dynamic int8 quant "
      "(SYCL extension)");
}
