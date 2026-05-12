#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <c10/core/DeviceGuard.h>

using namespace sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

template <typename decl_tag> struct QuantMax;
template <> struct QuantMax<int8_t> { static constexpr float value = 127.0f; };
template <> struct QuantMax<uint8_t> { static constexpr float value = 448.0f; }; // e4m3fn max value

template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;

    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    simd<uint8_t, N> res = 0;
    auto is_normal = (exp > 0) & (exp < 16);
    auto is_overflow = exp >= 16;
    auto is_underflow = exp <= 0;

    res.merge(sign | (exp << 3) | mantissa, is_normal);
    res.merge(sign | 0x7E, is_overflow);
    res.merge(sign, is_underflow);
    return res;
}

template <typename T_in, typename T_out>
void swiglu_dynamic_quant_impl(
    torch::Tensor& hidden_states,
    torch::Tensor& smooth_scale,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale) {

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    auto smooth_scale_ptr = smooth_scale.data_ptr<float>();
    auto quant_tokens_ptr = reinterpret_cast<T_out*>(quant_tokens.data_ptr());
    auto per_token_scale_ptr = per_token_scale.data_ptr<float>();

    int num_tokens = hidden_states.size(0);
    int hidden_size = hidden_states.size(1) / 2;
    constexpr float quant_max = QuantMax<T_out>::value;

    auto launch_swiglu = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hidden_size / BS;
        int wg_size = std::min(num_blocks, 64);

        sycl::range<2> GlobalRange(num_tokens, wg_size);
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                slm_init(1056);

                int token_idx = item.get_global_id(0);
                int tid = item.get_local_id(1);

                auto in_row_ptr = hidden_states_ptr + token_idx * (hidden_size * 2);
                auto out_row_ptr = quant_tokens_ptr + token_idx * hidden_size;
                uint32_t reduction_base = hidden_size * sizeof(float);

                simd<float, CHUNK> thread_max_vec = 0.0f;

                for (int bid = tid; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(in_row_ptr + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(in_row_ptr + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + bid * BS + u * CHUNK);
                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> scaled = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled));
                    }
                }

                float local_max = hmax<float, float, CHUNK>(thread_max_vec);
                slm_block_store<float, 4>(tid * 16, simd<float, 4>(local_max));
                barrier();

                if (tid == 0) {
                    float token_max = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        float thread_m = slm_block_load<float, 4>(i * 16)[0];
                        if (thread_m > token_max) token_max = thread_m;
                    }
                    float token_scale_val = token_max / quant_max;
                    token_scale_val = (token_scale_val == 0.0f) ? 1.0f : token_scale_val;
                    per_token_scale_ptr[token_idx] = token_scale_val;
                    slm_block_store<float, 4>(1024, simd<float, 4>(1.0f / token_scale_val));
                }
                barrier();
                float inv_scale = slm_block_load<float, 4>(1024)[0];

                for (int bid = tid; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(in_row_ptr + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(in_row_ptr + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + bid * BS + u * CHUNK);
                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> scaled = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        
                        simd<T_out, CHUNK> quantized_out;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized_out = rnde<float>(scaled * inv_scale);
                        } else {
                            quantized_out = fast_cvt_float_to_e4m3fn<CHUNK>(scaled * inv_scale);
                        }
                        block_store<T_out, CHUNK>(out_row_ptr + bid * BS + u * CHUNK, quantized_out);
                    }
                }
            });
        });
    };

    int target_total_threads = 1024;
    auto is_valid_unroll = [&](int unroll) {
        int bs = unroll * 64;
        int max_wg_size = std::min(hidden_size / bs, 64);
        return (hidden_size % bs == 0) && ((num_tokens * max_wg_size) >= target_total_threads);
    };

    int best_unroll = 1;

    for (int unroll : {32, 16, 8, 4, 2, 1}) {
        if (is_valid_unroll(unroll)) {
            best_unroll = unroll;
            break;
        }
    }

    switch (best_unroll) {
        case 32:  return launch_swiglu(std::integral_constant<int, 32>{});
        case 16:  return launch_swiglu(std::integral_constant<int, 16>{});
        case 8:  return launch_swiglu(std::integral_constant<int, 8>{});
        case 4:  return launch_swiglu(std::integral_constant<int, 4>{});
        case 2:  return launch_swiglu(std::integral_constant<int, 2>{});
        default: return launch_swiglu(std::integral_constant<int, 1>{});
    }
}

#define DISPATCH_QUANT_IMPL(FUNC_NAME, ...) \
    if (in_dtype == at::ScalarType::BFloat16 && out_dtype == at::ScalarType::Char) { \
        FUNC_NAME<bf16, int8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Char) { \
        FUNC_NAME<fp16, int8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::BFloat16 && out_dtype == at::ScalarType::Float8_e4m3fn) { \
        FUNC_NAME<bf16, uint8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Float8_e4m3fn) { \
        FUNC_NAME<fp16, uint8_t>(__VA_ARGS__); \
    } else { \
        TORCH_CHECK(false, "Unsupported in_dtype/out_dtype combination."); \
    }

void swiglu_dynamic_quant(
    torch::Tensor& hidden_states,
    torch::Tensor& smooth_scale,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale) {

    at::DeviceGuard guard(hidden_states.device());

    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
    TORCH_CHECK(quant_tokens.is_contiguous(), "quant_tokens must be contiguous");

    auto in_dtype = hidden_states.scalar_type();
    auto out_dtype = quant_tokens.scalar_type();

    DISPATCH_QUANT_IMPL(swiglu_dynamic_quant_impl,
                        hidden_states, smooth_scale, quant_tokens, per_token_scale);
}

PYBIND11_MODULE(swiglu_dynamic_quant_sycl, m) {
    m.def("swiglu_dynamic_quant", &swiglu_dynamic_quant, "SwiGLU Dynamic Quant");
}