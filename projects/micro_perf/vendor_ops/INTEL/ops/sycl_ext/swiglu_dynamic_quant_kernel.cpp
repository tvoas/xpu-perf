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

    auto launch_swiglu = [&](auto unroll_tag, auto slm_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr uint32_t SLM_BYTES = decltype(slm_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hidden_size / BS;
        int wg_size = std::min(num_blocks, 64);

        sycl::range<2> GlobalRange(num_tokens, wg_size);
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                int token_idx = item.get_global_id(0);
                int tid = item.get_local_id(1);

                slm_init(SLM_BYTES);

                auto in_row_ptr = hidden_states_ptr + token_idx * (hidden_size * 2);
                auto out_row_ptr = quant_tokens_ptr + token_idx * hidden_size;
                uint32_t reduction_base = hidden_size * sizeof(float); // Safely offset reduction array

                simd<float, CHUNK> thread_max_vec = 0.0f;

                // Pass 1: Read, compute, track extrema, and cache to SLM
                for (int bid = tid; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(
                            in_row_ptr + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(
                            in_row_ptr + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + bid * BS + u * CHUNK);
                        
                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> scaled = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled));
                        
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            slm_block_store<float, 16>(base_offset + i * 64, scaled.template select<16, 1>(i * 16));
                        }
                    }
                }

                float local_max = hmax<float, float, CHUNK>(thread_max_vec);

                slm_block_store<float, 4>(reduction_base + tid * 16, simd<float, 4>(local_max));
                barrier();

                float inv_scale = 1.0f;
                // WG-level reduction for global max
                if (tid == 0) {
                    float token_max = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        float thread_m = slm_block_load<float, 4>(reduction_base + i * 16)[0];
                        if (thread_m > token_max) token_max = thread_m;
                    }
                    float token_scale_val = token_max / quant_max;
                    token_scale_val = (token_scale_val == 0.0f) ? 1.0f : token_scale_val;
                    per_token_scale_ptr[token_idx] = token_scale_val;
                    slm_block_store<float, 4>(reduction_base + 1024, simd<float, 4>(1.0f / token_scale_val)); // inverse scale
                }
                barrier();
                inv_scale = slm_block_load<float, 4>(reduction_base + 1024)[0];

                // Pass 2: Quantize and write
                for (int bid = tid; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
                        simd<float, CHUNK> cached_tokens;

#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            cached_tokens.template select<16, 1>(i * 16) = slm_block_load<float, 16>(base_offset + i * 64);
                        }

                        simd<T_out, CHUNK> quantized_out;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized_out = rnde<float>(cached_tokens * inv_scale);
                        } else { // FP8 e4m3fn
                            quantized_out = fast_cvt_float_to_e4m3fn<CHUNK>(cached_tokens * inv_scale);
                        }

                        block_store<T_out, CHUNK>(out_row_ptr + bid * BS + u * CHUNK, quantized_out);
                    }
                }
            });
        });
    };

    auto dispatch_slm = [&](auto unroll_tag) {
        if (hidden_size <=   128) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   2560>{}); //   128 * 4 + 2048
        if (hidden_size <=   256) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   3072>{}); //   256 * 4 + 2048
        if (hidden_size <=   512) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   4096>{}); //   512 * 4 + 2048
        if (hidden_size <=  1024) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   6144>{}); //  1024 * 4 + 2048
        if (hidden_size <=  2048) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  10240>{}); //  2048 * 4 + 2048
        if (hidden_size <=  3072) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  14336>{}); //  3072 * 4 + 2048
        if (hidden_size <=  4096) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  18432>{}); //  4096 * 4 + 2048
        if (hidden_size <=  5120) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  22528>{}); //  5120 * 4 + 2048
        if (hidden_size <=  6144) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  26624>{}); //  6144 * 4 + 2048
        if (hidden_size <=  7168) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  30720>{}); //  7168 * 4 + 2048
        if (hidden_size <=  8192) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  34816>{}); //  8192 * 4 + 2048
        if (hidden_size <= 10240) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  43008>{}); // 10240 * 4 + 2048
        if (hidden_size <= 12288) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  51200>{}); // 12288 * 4 + 2048
        if (hidden_size <= 14336) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  59392>{}); // 14336 * 4 + 2048
        if (hidden_size <= 16384) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  67584>{}); // 16384 * 4 + 2048
        if (hidden_size <= 20480) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  83968>{}); // 20480 * 4 + 2048
        if (hidden_size <= 24576) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t, 100352>{}); // 24576 * 4 + 2048
        if (hidden_size <= 28672) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t, 116736>{}); // 28672 * 4 + 2048
                                  return launch_swiglu(unroll_tag, std::integral_constant<uint32_t, 131072>{}); // 32256 * 4 + 2048
    };

    int target_wg = 4;
    int num_chunks = hidden_size / 64;
    int best_unroll = 1;

    for (int u : {4, 2}) {
        if (num_chunks % u == 0 && (num_chunks / u) >= target_wg) {
            best_unroll = u;
            break;
        }
    }

    switch (best_unroll) {
        case 32: return dispatch_slm(std::integral_constant<int, 32>{});
        case 16: return dispatch_slm(std::integral_constant<int, 16>{});
        case 8:  return dispatch_slm(std::integral_constant<int, 8>{});
        case 4:  return dispatch_slm(std::integral_constant<int, 4>{});
        case 2:  return dispatch_slm(std::integral_constant<int, 2>{});
        default: return dispatch_slm(std::integral_constant<int, 1>{});
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