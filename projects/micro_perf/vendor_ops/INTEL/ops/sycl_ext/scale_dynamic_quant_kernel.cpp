// Symmetric per-token dynamic quantization with a per-channel
// `smooth_scale` pre-multiplier. SYCL extension for xpu-perf.
//
// Supported output dtypes: int8 / float8_e4m3fn / float8_e5m2.
//
//   smoothed[m, k] = (float)input[m, k] * smooth_scale[k]
//   amax[m]        = max_k |smoothed[m, k]|
//   per_token_scale[m] = amax[m] / kMax_out          // 127 / 448 / 57344
//   out[m, k]      = (smoothed[m, k] / per_token_scale[m])
//                        .clamp(-kMax_out, kMax_out)
//                        .to(out_dtype)              // int8 uses RNE rint
//
// Equivalent to torch.ops.torch_ipex.scale_dynamic_quant for int8;
// matches `smooth_per_token_dynamic_quant(..., dst=float8_e4m3fn)` for fp8.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

template <typename scalar_t>
struct vec4_t {
  scalar_t val[4];
};

// Generic 4 x 1-byte packed vector (32-bit aligned store).
// Used for int8 / Float8_e4m3fn / Float8_e5m2 outputs alike.
template <typename out_t>
struct alignas(4) b8x4_t {
  out_t v[4];
};

// Per-output-dtype quantization traits: clip range and float -> out conversion.
template <typename out_t>
struct quant_traits;

template <>
struct quant_traits<int8_t> {
  static constexpr float kMax = 127.0f;
  static inline int8_t convert(float r) {
    r = sycl::fmin(sycl::fmax(r, -kMax), kMax);
    return static_cast<int8_t>(sycl::rint(r));
  }
};

template <>
struct quant_traits<c10::Float8_e4m3fn> {
  static constexpr float kMax = 448.0f;
  static inline c10::Float8_e4m3fn convert(float r) {
    r = sycl::fmin(sycl::fmax(r, -kMax), kMax);
    // Float8_e4m3fn(float) is C10_HOST_DEVICE; uses RNE in software.
    return c10::Float8_e4m3fn(r);
  }
};

template <>
struct quant_traits<c10::Float8_e5m2> {
  static constexpr float kMax = 57344.0f;
  static inline c10::Float8_e5m2 convert(float r) {
    r = sycl::fmin(sycl::fmax(r, -kMax), kMax);
    return c10::Float8_e5m2(r);
  }
};

template <typename scalar_t, typename out_t>
class scale_dynamic_quant_kernel {
 private:
  out_t* out_;
  float* per_token_scale_;
  scalar_t const* input_;
  float const* smooth_scale_;
  int const hidden_size_;

 public:
  scale_dynamic_quant_kernel(
      out_t* out,
      float* per_token_scale,
      scalar_t const* input,
      float const* smooth_scale,
      int hidden_size)
      : out_(out),
        per_token_scale_(per_token_scale),
        input_(input),
        smooth_scale_(smooth_scale),
        hidden_size_(hidden_size) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const range = item.get_local_range(0);
    int const token_idx = item.get_group(0);

    int64_t const offset = static_cast<int64_t>(token_idx) * hidden_size_;
    scalar_t const* token_in = input_ + offset;
    out_t* token_out = out_ + offset;

    bool const can_vec = (hidden_size_ % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(token_in) & 7u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(smooth_scale_) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(token_out) & 3u) == 0u);

    using xvec_t = vec4_t<scalar_t>;
    using svec_t = vec4_t<float>;
    using ovec_t = b8x4_t<out_t>;
    auto const* vec_in = reinterpret_cast<xvec_t const*>(token_in);
    auto const* vec_s = reinterpret_cast<svec_t const*>(smooth_scale_);
    auto* vec_out = reinterpret_cast<ovec_t*>(token_out);
    int const num_vec = hidden_size_ >> 2;

    // Pass 1: row absmax of (input * smooth_scale)
    float thread_amax = 0.0f;
    if (can_vec) {
#pragma unroll 4
      for (int i = tid; i < num_vec; i += range) {
        xvec_t xv = vec_in[i];
        svec_t sv = vec_s[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float v = static_cast<float>(xv.val[j]) * sv.val[j];
          thread_amax = sycl::max(thread_amax, sycl::fabs(v));
        }
      }
    } else {
      for (int i = tid; i < hidden_size_; i += range) {
        float v = static_cast<float>(token_in[i]) * smooth_scale_[i];
        thread_amax = sycl::max(thread_amax, sycl::fabs(v));
      }
    }

    float const row_amax = sycl::reduce_over_group(
        item.get_group(), thread_amax, sycl::maximum<float>());

    auto& shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2]>(item.get_group());
    if (tid == 0) {
      constexpr float kMax = quant_traits<out_t>::kMax;
      float const scale = row_amax / kMax;
      shared[0] = scale;
      shared[1] = (row_amax > 0.0f) ? (kMax / row_amax) : 0.0f;
      per_token_scale_[token_idx] = scale;
    }
    group_barrier(item.get_group());
    float const inv_scale = shared[1];

    // Pass 2: quantize and store (4 x 1B packed -> single 32-bit store)
    if (can_vec) {
#pragma unroll 4
      for (int i = tid; i < num_vec; i += range) {
        xvec_t xv = vec_in[i];
        svec_t sv = vec_s[i];
        ovec_t ov;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float r = static_cast<float>(xv.val[j]) * sv.val[j] * inv_scale;
          ov.v[j] = quant_traits<out_t>::convert(r);
        }
        vec_out[i] = ov;
      }
    } else {
      for (int i = tid; i < hidden_size_; i += range) {
        float r =
            static_cast<float>(token_in[i]) * smooth_scale_[i] * inv_scale;
        token_out[i] = quant_traits<out_t>::convert(r);
      }
    }
  }
};

template <typename scalar_t, typename out_t>
static inline void launch_kernel(
    sycl::queue& queue,
    sycl::range<1> grid,
    sycl::range<1> block,
    at::Tensor& out,
    at::Tensor& per_token_scale,
    at::Tensor const& input,
    at::Tensor const& smooth_scale,
    int hidden_size) {
  queue.submit([&](sycl::handler& cgh) {
    auto kernel = scale_dynamic_quant_kernel<scalar_t, out_t>(
        reinterpret_cast<out_t*>(out.data_ptr()),
        per_token_scale.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        smooth_scale.data_ptr<float>(),
        hidden_size);
    cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
  });
}

void scale_dynamic_quant_forward(
    at::Tensor const& input,
    at::Tensor const& smooth_scale,
    at::Tensor& out,
    at::Tensor& per_token_scale) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dim");
  TORCH_CHECK(out.sizes() == input.sizes(), "out shape must match input");
  auto const out_dtype = out.scalar_type();
  TORCH_CHECK(
      out_dtype == at::kChar || out_dtype == at::kFloat8_e4m3fn ||
          out_dtype == at::kFloat8_e5m2,
      "out must be int8 / float8_e4m3fn / float8_e5m2");
  TORCH_CHECK(
      smooth_scale.scalar_type() == at::kFloat, "smooth_scale must be fp32");
  TORCH_CHECK(
      per_token_scale.scalar_type() == at::kFloat,
      "per_token_scale must be fp32");

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  TORCH_CHECK(
      smooth_scale.numel() == hidden_size,
      "smooth_scale numel must equal hidden_size");
  TORCH_CHECK(
      per_token_scale.numel() == num_tokens,
      "per_token_scale numel must equal num_tokens");

  if (num_tokens == 0) {
    return;
  }

  int local_size = std::min(hidden_size, 512);
  if (local_size > 32) {
    local_size = (local_size / 32) * 32;
  }
  sycl::range<1> grid(static_cast<size_t>(num_tokens));
  sycl::range<1> block(static_cast<size_t>(local_size));

  auto& queue = c10::xpu::getCurrentXPUStream(input.device().index()).queue();

#define DISPATCH_OUT(SCALAR_T)                                              \
  do {                                                                      \
    switch (out_dtype) {                                                    \
      case at::kChar:                                                       \
        launch_kernel<SCALAR_T, int8_t>(                                    \
            queue, grid, block, out, per_token_scale, input,                \
            smooth_scale, hidden_size);                                     \
        break;                                                              \
      case at::kFloat8_e4m3fn:                                              \
        launch_kernel<SCALAR_T, c10::Float8_e4m3fn>(                        \
            queue, grid, block, out, per_token_scale, input,                \
            smooth_scale, hidden_size);                                     \
        break;                                                              \
      case at::kFloat8_e5m2:                                                \
        launch_kernel<SCALAR_T, c10::Float8_e5m2>(                          \
            queue, grid, block, out, per_token_scale, input,                \
            smooth_scale, hidden_size);                                     \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false, "unsupported out dtype");                        \
    }                                                                       \
  } while (0)

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "scale_dynamic_quant_sycl_ext",
      AT_DISPATCH_CASE(
          at::ScalarType::Half, [&] { DISPATCH_OUT(scalar_t); })
          AT_DISPATCH_CASE(
              at::ScalarType::BFloat16, [&] { DISPATCH_OUT(scalar_t); })
              AT_DISPATCH_CASE(
                  at::ScalarType::Float, [&] { DISPATCH_OUT(scalar_t); }));
#undef DISPATCH_OUT
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "scale_dynamic_quant_forward",
      &scale_dynamic_quant_forward,
      "Symmetric per-token int8 dynamic quant with smooth_scale "
      "(SYCL extension)");
}
