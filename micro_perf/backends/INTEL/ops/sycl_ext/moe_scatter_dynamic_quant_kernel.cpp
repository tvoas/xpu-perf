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

// Vectorized software emulation for float32 -> float8_e4m3fn conversion
template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;

    // Rounding tie-to-even approximation (add half of the 20-bit shifted fractional part)
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;

    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    simd<uint8_t, N> res = 0;

    auto is_normal = (exp > 0) & (exp < 16);
    auto is_overflow = exp >= 16;
    auto is_underflow = exp <= 0;

    res.merge(sign | (exp << 3) | mantissa, is_normal);
    res.merge(sign | 0x7E, is_overflow); // 0x7E is 448.0 (Maximum e4m3fn value)
    res.merge(sign, is_underflow);       // Fast fallback: flush subnormals to 0

    return res;
}

template <typename T_in, typename T_out>
void moe_scatter_dynamic_quant_impl(
    torch::Tensor& selected_experts,
    torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale,
    torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale,
    torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    int n_tokens = selected_experts.size(0);
    if (n_tokens <= 0) {
        return;
    }

    // Python frontend concatenates shared experts into selected_experts,
    // meaning the XPU kernel processes them uniformly via the topk dimension.
    (void)shared_experts_num;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    int topk = selected_experts.size(1);
    int n_expert = experts_token_count.size(0);
    int hd_size = hidden_states.size(1);
    int n_expert_total = n_expert;
    constexpr int EXPERT_TILE = 8;

    constexpr float quant_max = QuantMax<T_out>::value;

    auto selected_experts_ptr = selected_experts.data_ptr<int32_t>();
    auto ext_tokens_cnt_ptr = experts_token_count.data_ptr<int32_t>();
    auto ext_tokens_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto token_to_scatter_offset_ptr = token_to_scatter_offset.data_ptr<int32_t>();

    // UNIFIED ROUTING PASS: Replaces Pass 0, Pass 1, and Pass 2, eliminating two
    // kernel launch overheads (~15us+ saved) and all global atomics.
    auto routing_event = queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> local_expert_counts(n_expert_total, cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
            int lid = item.get_local_id(0);

            // 1. Zero out SLM histogram
            for (int i = lid; i < n_expert_total; i += 256) {
                local_expert_counts[i] = 0;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // 2. Count tokens using rapid SLM atomics
            int total_items = n_tokens * topk;
            for (int i = lid; i < total_items; i += 256) {
                int expert_id = selected_experts_ptr[i];
                if (expert_id >= 0 && expert_id < n_expert_total) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                        atomic_cnt(local_expert_counts[expert_id]);
                    // Offset uniquely maps to bin density
                    token_to_scatter_offset_ptr[i] = atomic_cnt.fetch_add(1);
                } else {
                    token_to_scatter_offset_ptr[i] = 0;
                }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // 3. Prefix sum & Export to Global Memory
            // (A sequential loop inside a single thread is dramatically faster
            // than launching a separate kernel for small expert counts)
            if (lid == 0) {
                int32_t sum = 0;
                for (int i = 0; i < n_expert_total; ++i) {
                    int count = local_expert_counts[i];
                    ext_tokens_cnt_ptr[i] = count;
                    ext_tokens_start_ptr[i] = sum;
                    sum += count;
                }
            }
        });
    });

    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    auto smooth_scale_ptr = experts_smooth_scale.data_ptr<float>();
    auto moe_weights_ptr = moe_weights.data_ptr<float>();
    auto scatter_tokens_ptr = reinterpret_cast<T_out*>(scatter_tokens.data_ptr());
    auto scatter_per_token_scale_ptr = scatter_per_token_scale.data_ptr<float>();
    auto scatter_tokens_offset_ptr = scatter_tokens_offset.data_ptr<int32_t>();

    // Legacy per-token-expert scatter path keeps stronger mean throughput once work scales.
    auto launch_scatter_legacy = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hd_size / BS;
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tokens * topk, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {

                const int token_k_idx = item.get_group(0);
                const int expert_id = selected_experts_ptr[token_k_idx];

                if (expert_id < 0 || expert_id >= n_expert_total) {
                    return;
                }

                slm_init(1056);

                const int loc_id = item.get_local_id(1);

                const int token_idx = token_k_idx / topk;

                const int offset = token_to_scatter_offset_ptr[token_k_idx];
                const int expert_start = ext_tokens_start_ptr[expert_id];

                const int target_idx = expert_start + offset;
                const float weight = moe_weights_ptr[token_k_idx];

                simd<float, CHUNK> thread_max_vec = 0.0f;

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        simd<float, CHUNK> smoothed_abs = sycl::ext::intel::esimd::abs(smoothed);

                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, smoothed_abs);
                    }
                }

                float thread_max = hmax<float, float, CHUNK>(thread_max_vec);

                slm_block_store<float, 4>(loc_id * 16, simd<float, 4>(thread_max));
                barrier();

                float this_token_scale = 1.0f;

                if (loc_id == 0) {
                    float max_value_final = 0.0f;

                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(i * 16);
                        if (val[0] > max_value_final) max_value_final = val[0];
                    }

                    float raw_token_scale = max_value_final / quant_max;
                    this_token_scale = raw_token_scale == 0.0f ? 1.0f : raw_token_scale;

                    slm_block_store<float, 4>(1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(1024)[0];
                float recip_scale = 1.0f / this_token_scale;

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;

                        simd<T_out, CHUNK> quantized;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized = rnde<float>(smoothed * recip_scale);
                        } else {
                            quantized = fast_cvt_float_to_e4m3fn<CHUNK>(smoothed * recip_scale);
                        }

                        block_store<T_out, CHUNK>(
                            scatter_tokens_ptr + target_idx * hd_size + hd_bid * BS + u * CHUNK, quantized);
                    }
                }

                if (loc_id == 0) {
                    block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, this_token_scale);
                    block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, token_idx);
                }
            });
        });
    };

    // Grouped per-token scatter path improves some underfilled shapes by reusing hidden loads across experts.
    auto launch_scatter_grouped = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hd_size / BS;
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tokens, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                // Increase SLM to handle OWORD block alignment: 1056B
                slm_init(1056);

                const int token_idx = item.get_group(0);
                const int loc_id = item.get_local_id(1);

                const int token_base = token_idx * topk;

                for (int slot_base = 0; slot_base < topk; slot_base += EXPERT_TILE) {
                    const int tile_slots = (topk - slot_base) < EXPERT_TILE ? (topk - slot_base) : EXPERT_TILE;

                    int expert_ids[EXPERT_TILE];
                    int target_idxs[EXPERT_TILE];
                    float weights[EXPERT_TILE];
                    float thread_max[EXPERT_TILE];
                    float token_scales[EXPERT_TILE];

                    for (int slot = 0; slot < EXPERT_TILE; ++slot) {
                        expert_ids[slot] = -1;
                        target_idxs[slot] = 0;
                        weights[slot] = 0.0f;
                        thread_max[slot] = 0.0f;
                        token_scales[slot] = 1.0f;
                    }

                    for (int slot = 0; slot < tile_slots; ++slot) {
                        const int global_slot = token_base + slot_base + slot;
                        const int expert_id = selected_experts_ptr[global_slot];
                        if (expert_id < 0 || expert_id >= n_expert_total) {
                            continue;
                        }

                        expert_ids[slot] = expert_id;
                        weights[slot] = moe_weights_ptr[global_slot];

                        const int offset = token_to_scatter_offset_ptr[global_slot];
                        const int expert_start = ext_tokens_start_ptr[expert_id];
                        target_idxs[slot] = expert_start + offset;
                    }

                    // Pass 1: load each hidden-state chunk once and accumulate maxima for this expert tile.
                    for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                                hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                            simd<float, CHUNK> hidden_fp = simd<float, CHUNK>(hidden);

                            for (int slot = 0; slot < tile_slots; ++slot) {
                                const int expert_id = expert_ids[slot];
                                if (expert_id < 0) {
                                    continue;
                                }

                                simd<float, CHUNK> scale = block_load<float, CHUNK>(
                                    smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                                simd<float, CHUNK> smoothed = hidden_fp * scale * weights[slot];
                                float chunk_max = hmax<float, float, CHUNK>(
                                    sycl::ext::intel::esimd::abs(smoothed));

                                if (chunk_max > thread_max[slot]) {
                                    thread_max[slot] = chunk_max;
                                }
                            }
                        }
                    }

                    for (int slot = 0; slot < tile_slots; ++slot) {
                        if (expert_ids[slot] < 0) {
                            continue;
                        }

                        // Use SLM Block Stores array spacing for max bandwidth (16-byte aligned natively)
                        slm_block_store<float, 4>(loc_id * 16, simd<float, 4>(thread_max[slot]));
                        barrier();

                        if (loc_id == 0) {
                            float max_value_final = 0.0f;

                            for (int i = 0; i < wg_size; i++) {
                                simd<float, 4> val = slm_block_load<float, 4>(i * 16);
                                if (val[0] > max_value_final) max_value_final = val[0];
                            }

                            float raw_token_scale = max_value_final / quant_max;
                            float this_token_scale = raw_token_scale == 0.0f ? 1.0f : raw_token_scale;

                            slm_block_store<float, 4>(1024, simd<float, 4>(this_token_scale));
                        }
                        barrier();

                        token_scales[slot] = slm_block_load<float, 4>(1024)[0];
                        barrier();
                    }

                    // Pass 2: reuse each hidden-state chunk across the same expert tile for quantization.
                    for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                                hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                            simd<float, CHUNK> hidden_fp = simd<float, CHUNK>(hidden);

                            for (int slot = 0; slot < tile_slots; ++slot) {
                                const int expert_id = expert_ids[slot];
                                if (expert_id < 0) {
                                    continue;
                                }

                                simd<float, CHUNK> scale = block_load<float, CHUNK>(
                                    smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);
                                simd<float, CHUNK> smoothed = hidden_fp * scale * weights[slot];
                                float recip_scale = 1.0f / token_scales[slot];

                                simd<T_out, CHUNK> quantized;
                                if constexpr (std::is_same_v<T_out, int8_t>) {
                                    quantized = rnde<float>(smoothed * recip_scale);
                                } else {
                                    quantized = fast_cvt_float_to_e4m3fn<CHUNK>(smoothed * recip_scale);
                                }

                                block_store<T_out, CHUNK>(
                                    scatter_tokens_ptr + target_idxs[slot] * hd_size + hd_bid * BS + u * CHUNK, quantized);
                            }
                        }
                    }

                    if (loc_id == 0) {
                        for (int slot = 0; slot < tile_slots; ++slot) {
                            if (expert_ids[slot] < 0) {
                                continue;
                            }

                            block_store<float, 1>(scatter_per_token_scale_ptr + target_idxs[slot], token_scales[slot]);
                            block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idxs[slot], token_idx);
                        }
                    }
                }
            });
        });
    };

    int total_scatter_items = n_tokens * topk;
    int total_token_groups = n_tokens;

    int target_total_threads = 1024;

    auto is_valid_unroll_legacy = [&](int unroll) {
        int bs = unroll * 64;
        int max_wg_size = std::min(hd_size / bs, 64);
        return (hd_size % bs == 0) && ((total_scatter_items * max_wg_size) >= target_total_threads);
    };

    auto is_valid_unroll_grouped = [&](int unroll) {
        int bs = unroll * 64;
        int max_wg_size = std::min(hd_size / bs, 64);
        return (hd_size % bs == 0) && ((total_token_groups * max_wg_size) >= target_total_threads);
    };

    // V06/V07 data shows the grouped path helps underfilled regimes, while the legacy path
    // retains better mean throughput once token count, top-k, or hidden size grows.
    bool prefer_grouped =
        (hd_size <= 512 && n_tokens <= 512) ||
        (hd_size <= 2048 && n_expert_total <= 128 &&
            ((topk <= 2 && n_tokens <= 1024) ||
             (topk <= 5 && n_tokens <= 512) ||
             (topk > 5 && n_tokens <= 128)));

    if (prefer_grouped) {
        if (is_valid_unroll_grouped(32)) return launch_scatter_grouped(std::integral_constant<int, 32>{});
        if (is_valid_unroll_grouped(16)) return launch_scatter_grouped(std::integral_constant<int, 16>{});
        if (is_valid_unroll_grouped(8))  return launch_scatter_grouped(std::integral_constant<int, 8>{});
        if (is_valid_unroll_grouped(4))  return launch_scatter_grouped(std::integral_constant<int, 4>{});
        if (is_valid_unroll_grouped(2))  return launch_scatter_grouped(std::integral_constant<int, 2>{});
                                        return launch_scatter_grouped(std::integral_constant<int, 1>{});
    }

    if (is_valid_unroll_legacy(32)) return launch_scatter_legacy(std::integral_constant<int, 32>{});
    if (is_valid_unroll_legacy(16)) return launch_scatter_legacy(std::integral_constant<int, 16>{});
    if (is_valid_unroll_legacy(8))  return launch_scatter_legacy(std::integral_constant<int, 8>{});
    if (is_valid_unroll_legacy(4))  return launch_scatter_legacy(std::integral_constant<int, 4>{});
    if (is_valid_unroll_legacy(2))  return launch_scatter_legacy(std::integral_constant<int, 2>{});
                                    return launch_scatter_legacy(std::integral_constant<int, 1>{});
}

// Outer dispatch macros to select implementation
#define DISPATCH_MOE_QUANT_IMPL(FUNC_NAME, ...) \
    if (in_dtype == at::ScalarType::BFloat16 && out_dtype == at::ScalarType::Char) { \
        FUNC_NAME<bf16, int8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Char) { \
        FUNC_NAME<fp16, int8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::BFloat16 && out_dtype == at::ScalarType::Float8_e4m3fn) { \
        FUNC_NAME<bf16, uint8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Float8_e4m3fn) { \
        FUNC_NAME<fp16, uint8_t>(__VA_ARGS__); \
    } else { \
        TORCH_CHECK(false, "Unsupported in_dtype/out_dtype combination for dynamic quant."); \
    }

void moe_scatter_dynamic_quant(
    torch::Tensor& selected_experts,
    torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale,
    torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale,
    torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    at::DeviceGuard guard(hidden_states.device());

    // Contiguity checks
    TORCH_CHECK(selected_experts.is_contiguous(), "selected_experts must be contiguous");
    TORCH_CHECK(moe_weights.is_contiguous(), "moe_weights must be contiguous");
    TORCH_CHECK(token_to_scatter_offset.is_contiguous(), "token_to_scatter_offset must be contiguous");
    TORCH_CHECK(experts_token_count.is_contiguous(), "experts_token_count must be contiguous");
    TORCH_CHECK(experts_token_start.is_contiguous(), "experts_token_start must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(experts_smooth_scale.is_contiguous(), "experts_smooth_scale must be contiguous");
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    TORCH_CHECK(scatter_per_token_scale.is_contiguous(), "scatter_per_token_scale must be contiguous");
    TORCH_CHECK(scatter_tokens_offset.is_contiguous(), "scatter_tokens_offset must be contiguous");

    // Dtype checks (Int32)
    TORCH_CHECK(selected_experts.scalar_type() == at::ScalarType::Int, "selected_experts must be Int32");
    TORCH_CHECK(token_to_scatter_offset.scalar_type() == at::ScalarType::Int, "token_to_scatter_offset must be Int32");
    TORCH_CHECK(experts_token_count.scalar_type() == at::ScalarType::Int, "experts_token_count must be Int32");
    TORCH_CHECK(experts_token_start.scalar_type() == at::ScalarType::Int, "experts_token_start must be Int32");
    TORCH_CHECK(scatter_tokens_offset.scalar_type() == at::ScalarType::Int, "scatter_tokens_offset must be Int32");

    // Dtype checks (Float32)
    TORCH_CHECK(moe_weights.scalar_type() == at::ScalarType::Float, "moe_weights must be Float32");
    TORCH_CHECK(experts_smooth_scale.scalar_type() == at::ScalarType::Float, "experts_smooth_scale must be Float32");
    TORCH_CHECK(scatter_per_token_scale.scalar_type() == at::ScalarType::Float, "scatter_per_token_scale must be Float32");

    // Logical shape checks
    TORCH_CHECK(experts_token_count.size(0) == experts_token_start.size(0), "Token count and start tensors must match in size");

    // Block size alignment check for ESIMD vectorized loads
    int64_t hd_size = hidden_states.size(1);
    TORCH_CHECK(hd_size >= 64 && hd_size % 64 == 0,
                "hidden_states inner dimension must be a positive multiple of 64 for XPU block loads, got ", hd_size);

    auto in_dtype = hidden_states.scalar_type();
    auto out_dtype = scatter_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_scatter_dynamic_quant_impl,
                            selected_experts, moe_weights, token_to_scatter_offset,
                            experts_token_count, experts_token_start, hidden_states,
                            experts_smooth_scale, scatter_tokens, scatter_per_token_scale,
                            scatter_tokens_offset, shared_experts_num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_scatter_dynamic_quant", &moe_scatter_dynamic_quant, "MoE Scatter Dynamic Quant");
}
