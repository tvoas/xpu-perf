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

/*
 * MoE scatter + per-token dynamic quantization.
 *
 * This kernel performs two jobs that are often separate in higher-level code:
 *   1. routing/scatter bookkeeping for token -> expert assignment
 *   2. per-routed-row dynamic quantization of the token payload
 *
 * Why that is useful in an MoE model:
 *   - Each token may be sent to one or more experts (top-k routing).
 *   - Before an expert-local GEMM can run efficiently, tokens must be grouped
 *     by expert into contiguous segments.
 *   - This kernel builds that grouped layout and writes the quantized payload in
 *     the same pass sequence, reducing intermediate traffic.
 *
 * Contract:
 *   - selected_experts / moe_weights are shaped [num_tokens, topk].
 *     For token i and route j:
 *       selected_experts[i, j] = expert id or -1
 *       moe_weights[i, j]      = routing weight for that expert choice
 *   - The kernel itself produces routing workspaces:
 *       token_to_scatter_offset
 *           per (token, route) offset inside that expert's local segment
 *       experts_token_count
 *           number of routed rows assigned to each expert
 *       experts_token_start
 *           prefix-sum start offset of each expert segment in scatter order
 *   - hidden_states and experts_smooth_scale are then fused with the routing
 *     metadata to produce scatter_tokens, per-token scales, and source-token ids.
 *
 * High-level execution outline:
 *   1. A lightweight routing kernel builds a histogram of how many routed rows
 *      land on each expert and assigns a per-expert offset to every token-route
 *      pair.
 *   2. A second ESIMD kernel treats every valid token-route pair as one output
 *      row in expert-grouped order.
 *   3. That second kernel loads the source hidden state, applies expert smooth
 *      scale and moe routing weight, finds the row max, derives a dynamic scale,
 *      quantizes the row, and writes it into the expert-grouped output buffer.
 *
 * If you are new to MoE routing terminology:
 *   - "scatter" here means rearranging token rows into expert-major order.
 *   - "dispatch order" means the order after grouping by expert.
 *   - "source token id" is preserved separately so downstream stages can later
 *     relate a scattered row back to its original token.
 */

// Small type trait translating an output element type into the maximum absolute
// value used when deriving a dynamic quantization scale.
template <typename decl_tag> struct QuantMax;
// int8 uses 127 as the largest positive value in the symmetric range.
template <> struct QuantMax<int8_t> { static constexpr float value = 127.0f; };
// FP8 e4m3fn uses 448 as the largest finite magnitude.
template <> struct QuantMax<uint8_t> { static constexpr float value = 448.0f; }; // e4m3fn max value

// Vectorized software emulation for float32 -> float8_e4m3fn conversion
template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    // Software FP8 conversion used when the destination dtype is e4m3fn.
    // Raw float bits for each lane in the SIMD vector.
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    // Sign bit shifted into the destination FP8 layout position.
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    // Magnitude bits with the sign removed.
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;

    // Rounding tie-to-even approximation (add half of the 20-bit shifted fractional part)
    // Rounding bias applied before the mantissa truncation step.
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;

    // Float32 exponent rebased into the FP8 e4m3fn exponent domain.
    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    // Highest 3 mantissa bits kept for the destination format.
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    // Output vector initialized to zero before masked merge operations.
    simd<uint8_t, N> res = 0;

    // Mask selecting normally representable values.
    auto is_normal = (exp > 0) & (exp < 16);
    // Mask selecting values that must clamp high.
    auto is_overflow = exp >= 16;
    // Mask selecting values that flush to zero in this fast path.
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

    // Number of source token rows before top-k expansion.
    int n_tokens = selected_experts.size(0);
    if (n_tokens <= 0) {
        return;
    }

    // The Python wrapper may already have folded "shared experts" into the same
    // selected_experts table used for ordinary routed experts. By the time the
    // data reaches this kernel, there is no special-case logic left to do.
    // Every route entry is handled uniformly as one token-to-expert assignment.
    (void)shared_experts_num;

    // Use the current PyTorch XPU stream to preserve execution order with the
    // rest of the benchmark pipeline.
    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    // Number of expert choices stored per token row.
    int topk = selected_experts.size(1);
    // Number of experts represented by the output metadata tensors.
    int n_expert = experts_token_count.size(0);
    // Hidden width of each source token row.
    int hd_size = hidden_states.size(1);
    // Dense expert count used by the kernel after any wrapper-side remapping.
    int n_expert_total = n_expert;

    // Destination-type saturation point used to derive per-row scales.
    constexpr float quant_max = QuantMax<T_out>::value;

    // Base pointer for the token-by-topk expert id table.
    auto selected_experts_ptr = selected_experts.data_ptr<int32_t>();
    // Base pointer for the per-expert routed-row counts written by the routing pass.
    auto ext_tokens_cnt_ptr = experts_token_count.data_ptr<int32_t>();
    // Base pointer for the per-expert segment starts written by the routing pass.
    auto ext_tokens_start_ptr = experts_token_start.data_ptr<int32_t>();
    // Base pointer for the per-(token, route) local offsets written by the routing pass.
    auto token_to_scatter_offset_ptr = token_to_scatter_offset.data_ptr<int32_t>();

    // Routing pass.
    //
    // This first kernel does not touch hidden_states at all. Its only job is to
    // build the metadata that maps each (token, top-k choice) pair into a unique
    // position inside the final expert-grouped scatter buffer.
    //
    // The implementation intentionally uses a single work-group and local memory
    // because the number of experts is small in this microbenchmark and the cost
    // of a separate parallel prefix-sum kernel would be higher than a short
    // work-group-local histogram plus a tiny serial scan.
    auto routing_event = queue.submit([&](sycl::handler& cgh) {
        // Work-group-local histogram with one counter per expert.
        sycl::local_accessor<int32_t, 1> local_expert_counts(n_expert_total, cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
            // Linear local thread id inside the single work-group used by the routing pass.
            int lid = item.get_local_id(0);

            // 1. Zero the work-group-local histogram.
            // local_expert_counts lives in local memory, so all 256 threads can
            // initialize it cooperatively.
            for (int i = lid; i < n_expert_total; i += 256) {
                local_expert_counts[i] = 0;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // 2. Visit every (token, route) pair and count how many assignments
            // each expert receives.
            //
            // fetch_add returns the old count before incrementing, which is
            // exactly the dense per-expert offset we need for that pair.
            // Total number of token-route pairs that must be examined.
            int total_items = n_tokens * topk;
            for (int i = lid; i < total_items; i += 256) {
                // Expert id selected by the router for this flattened token-route slot.
                int expert_id = selected_experts_ptr[i];
                if (expert_id >= 0 && expert_id < n_expert_total) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                        atomic_cnt(local_expert_counts[expert_id]);
                    // Example:
                    //   if this is the 4th token-route pair seen for expert 7,
                    //   fetch_add returns 3, meaning this row will become the
                    //   4th row inside expert 7's contiguous scatter segment.
                    token_to_scatter_offset_ptr[i] = atomic_cnt.fetch_add(1);
                } else {
                    // Invalid expert ids are ignored by the later scatter pass.
                    token_to_scatter_offset_ptr[i] = 0;
                }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // 3. Convert the histogram into expert segment metadata.
            // experts_token_count[i] = how many rows expert i owns
            // experts_token_start[i] = where expert i's segment starts in the
            //                          final scatter buffer
            //
            // A single-thread prefix sum is sufficient here because the number
            // of experts is small. This is one of the few cases where a serial
            // loop on device is simpler and faster than another kernel launch.
            if (lid == 0) {
                // Running prefix sum that becomes the start offset of each expert segment.
                int32_t sum = 0;
                for (int i = 0; i < n_expert_total; ++i) {
                    // Histogram count for expert i accumulated by the work-group.
                    int count = local_expert_counts[i];
                    ext_tokens_cnt_ptr[i] = count;
                    ext_tokens_start_ptr[i] = sum;
                    sum += count;
                }
            }
        });
    });

    // Base pointer for the source hidden-state matrix.
    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    // Base pointer for the expert-specific smoothing matrix.
    auto smooth_scale_ptr = experts_smooth_scale.data_ptr<float>();
    // Base pointer for the routing weights associated with each token-route pair.
    auto moe_weights_ptr = moe_weights.data_ptr<float>();
    // Base pointer for the quantized expert-grouped scatter output.
    auto scatter_tokens_ptr = reinterpret_cast<T_out*>(scatter_tokens.data_ptr());
    // Base pointer for the per-scattered-row dequant scale output.
    auto scatter_per_token_scale_ptr = scatter_per_token_scale.data_ptr<float>();
    // Base pointer for the original token id recorded for each scattered row.
    auto scatter_tokens_offset_ptr = scatter_tokens_offset.data_ptr<int32_t>();

    // Gather + quantize + scatter pass.
    //
    // The routing kernel above established where each token-route pair belongs.
    // This second kernel now materializes the actual payload rows in that exact
    // layout.
    auto launch_scatter = [&](auto unroll_tag) {
        // Number of vector chunks processed per inner-loop iteration.
        constexpr int UNROLL = decltype(unroll_tag)::value;
        // ESIMD vector width used by the block-load path.
        constexpr int CHUNK = 64;
        // Hidden elements covered by one unrolled iteration.
        constexpr int BS = CHUNK * UNROLL;

        // hidden dimension work partitioning. Each work-group handles one routed
        // row, and the threads within the group split the hidden dimension.
        // Number of BS-sized blocks required to cover one token row.
        int num_blocks = hd_size / BS;
        // Cooperative thread count used by one work-group processing one routed row.
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tokens * topk, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {

                // token_k_idx is a flattened (token, top-k slot) index.
                // group(0) rather than global_id(0) is used because the local
                // range in dimension 0 is 1, so each work-group owns one routed
                // row candidate.
                // Flattened token-route index owned by this work-group.
                const int token_k_idx = item.get_group(0);
                // Expert id chosen by the router for that token-route pair.
                const int expert_id = selected_experts_ptr[token_k_idx];

                if (expert_id < 0 || expert_id >= n_expert_total) {
                    return;
                }

                // This kernel only needs a small SLM footprint because it does
                // not cache the whole row, only a reduction scratch region plus
                // one broadcast scale value. 1056 bytes is enough for up to 64
                // per-thread maxima (64 * 16 bytes) plus one extra slot.
                slm_init(1056);

                // Thread id inside the work-group that is collaborating on this row.
                const int loc_id = item.get_local_id(1);

                // Recover the original token row and the per-expert local offset
                // that the routing kernel assigned.
                // Original source-token row index recovered from the flattened token-route id.
                const int token_idx = token_k_idx / topk;

                // Local offset of this row within the chosen expert segment.
                const int offset = token_to_scatter_offset_ptr[token_k_idx];
                // Global starting row of the chosen expert segment.
                const int expert_start = ext_tokens_start_ptr[expert_id];

                // target_idx is the final row index in expert-grouped order.
                const int target_idx = expert_start + offset;
                // moe weight is the router probability/weight for this specific
                // token -> expert assignment.
                // Routing weight attached to this token -> expert assignment.
                const float weight = moe_weights_ptr[token_k_idx];

                // Running per-thread max over the row slices processed by this thread.
                simd<float, CHUNK> thread_max_vec = 0.0f;

                // Pass 1: read the source token row, apply expert smooth scale
                // and routing weight, and find the maximum absolute value across
                // the row. Unlike the SwiGLU kernels, this kernel does not need
                // to cache the full float row because recomputing the simple
                // multiply path is cheaper than reserving large SLM per group.
                // Cooperative thread-strided traversal over the hidden dimension.
                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    // Unroll inside each assigned block to increase throughput.
                    for (int u = 0; u < UNROLL; ++u) {
                        // Load the source hidden-state slice for this token row.
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        // Load the smooth-scale slice belonging to the selected expert.
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                        // Numerically, each routed row becomes:
                        //   hidden_states[token_idx] * experts_smooth_scale[expert_id] * moe_weight
                        // Float activation slice after applying expert smoothing and router weight.
                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        // Absolute magnitude used for row-wise dynamic scale discovery.
                        simd<float, CHUNK> smoothed_abs = sycl::ext::intel::esimd::abs(smoothed);

                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, smoothed_abs);
                    }
                }

                // Collapse the per-thread vector max to one scalar contribution.
                float thread_max = hmax<float, float, CHUNK>(thread_max_vec);

                // Store one scalar max per thread into SLM for the work-group
                // reduction. The 16-byte spacing matches the block load/store
                // primitive's natural alignment.
                slm_block_store<float, 4>(loc_id * 16, simd<float, 4>(thread_max));
                barrier();

                // Default row scale before thread 0 computes the true value.
                float this_token_scale = 1.0f;

                if (loc_id == 0) {
                    // Final row-wide max reduction.
                    // Accumulator for the row-wise maximum across all participating threads.
                    float max_value_final = 0.0f;

                    // Scan the SLM scratch array containing one max per thread.
                    for (int i = 0; i < wg_size; i++) {
                        // Only lane 0 contains the scalar max published by that thread.
                        simd<float, 4> val = slm_block_load<float, 4>(i * 16);
                        if (val[0] > max_value_final) max_value_final = val[0];
                    }

                    // Store the dequant scale for this scattered row. Later, a
                    // consumer can reconstruct approximate float values via
                    // quantized * this_token_scale.
                    float raw_token_scale = max_value_final / quant_max;
                    this_token_scale = raw_token_scale == 0.0f ? 1.0f : raw_token_scale;

                    // Publish the scale once so all threads can use it in pass 2.
                    slm_block_store<float, 4>(1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                // Reload the broadcast dequant scale from SLM.
                this_token_scale = slm_block_load<float, 4>(1024)[0];
                // Precompute the reciprocal because the quantization pass multiplies by it repeatedly.
                float recip_scale = 1.0f / this_token_scale;

                // Pass 2: recompute the scaled float values, divide by the row
                // scale, convert to the destination dtype, and write them into
                // the expert-grouped output location.
                // Second traversal over the same row slices, now performing the actual quantization.
                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    // Keep the same unrolled structure as pass 1.
                    for (int u = 0; u < UNROLL; ++u) {
                        // Reload the source hidden-state slice because this kernel chose recomputation over full-row SLM caching.
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        // Reload the matching expert smooth-scale slice.
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                        // Recompute the float activation slice that will be quantized.
                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;

                        // Output register holding the quantized slice.
                        simd<T_out, CHUNK> quantized;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized = rnde<float>(smoothed * recip_scale);
                        } else {
                            quantized = fast_cvt_float_to_e4m3fn<CHUNK>(smoothed * recip_scale);
                        }

                        // target_idx already accounts for expert segment start
                        // and this row's local offset inside that segment.
                        // Store the quantized slice into the expert-grouped output matrix.
                        block_store<T_out, CHUNK>(
                            scatter_tokens_ptr + target_idx * hd_size + hd_bid * BS + u * CHUNK, quantized);
                    }
                }

                if (loc_id == 0) {
                    // Besides the quantized row itself, downstream kernels need
                    // both the dequant scale and the original source token id.
                    block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, this_token_scale);
                    block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, token_idx);
                }
            });
        });
    };

    // Total number of flattened token-route rows considered by the scatter launch.
    int total_scatter_items = n_tokens * topk;

    // Small batches can become under-occupied if we choose a very wide unroll,
    // because wider unroll reduces the number of blocks per row. This heuristic
    // only enables aggressive unroll factors when the total amount of work still
    // leaves enough threads active across the whole launch.
    // Minimum total thread count we want the full launch to expose before accepting a wide unroll.
    int target_total_threads = 1024;

    // Helper deciding whether a candidate unroll factor leaves enough overall work in flight.
    auto is_valid_unroll = [&](int unroll) {
        // Elements covered by one inner-loop iteration under this candidate unroll.
        int bs = unroll * 64;
        // Maximum work-group width available after that larger block size is applied.
        int max_wg_size = std::min(hd_size / bs, 64);
        return (hd_size % bs == 0) && ((total_scatter_items * max_wg_size) >= target_total_threads);
    };

    // Prefer the widest clean unroll factor that preserves enough parallel work.
    if (is_valid_unroll(32)) return launch_scatter(std::integral_constant<int, 32>{});
    if (is_valid_unroll(16)) return launch_scatter(std::integral_constant<int, 16>{});
    if (is_valid_unroll(8))  return launch_scatter(std::integral_constant<int, 8>{});
    if (is_valid_unroll(4))  return launch_scatter(std::integral_constant<int, 4>{});
    if (is_valid_unroll(2))  return launch_scatter(std::integral_constant<int, 2>{});
                             return launch_scatter(std::integral_constant<int, 1>{});
}

// Outer dispatch macros to select implementation
// Macro translating runtime PyTorch dtypes into the corresponding C++ template pair.
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

    // Bind validation and launch to the hidden-state device selected by PyTorch.
    at::DeviceGuard guard(hidden_states.device());

    // Contiguity checks
    // selected_experts must be contiguous because the routing pass treats it as a flat dense table.
    TORCH_CHECK(selected_experts.is_contiguous(), "selected_experts must be contiguous");
    // moe_weights must be contiguous because each token-route pair reads one aligned weight entry.
    TORCH_CHECK(moe_weights.is_contiguous(), "moe_weights must be contiguous");
    // token_to_scatter_offset must be contiguous because the routing pass writes it densely in place.
    TORCH_CHECK(token_to_scatter_offset.is_contiguous(), "token_to_scatter_offset must be contiguous");
    // experts_token_count must be contiguous because the prefix-sum kernel writes it as a dense expert array.
    TORCH_CHECK(experts_token_count.is_contiguous(), "experts_token_count must be contiguous");
    // experts_token_start must be contiguous because the prefix-sum kernel writes one start per expert.
    TORCH_CHECK(experts_token_start.is_contiguous(), "experts_token_start must be contiguous");
    // hidden_states must be contiguous because the scatter pass performs vectorized block loads from it.
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    // experts_smooth_scale must be contiguous because each row slice is read via vectorized block loads.
    TORCH_CHECK(experts_smooth_scale.is_contiguous(), "experts_smooth_scale must be contiguous");
    // scatter_tokens must be contiguous because the quantized output is written as dense expert-grouped rows.
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    // scatter_per_token_scale must be contiguous because one scalar scale is written per scattered row.
    TORCH_CHECK(scatter_per_token_scale.is_contiguous(), "scatter_per_token_scale must be contiguous");
    // scatter_tokens_offset must be contiguous because one source-token id is written per scattered row.
    TORCH_CHECK(scatter_tokens_offset.is_contiguous(), "scatter_tokens_offset must be contiguous");

    // Dtype checks (Int32)
    // selected_experts stores integer expert ids or -1 sentinels.
    TORCH_CHECK(selected_experts.scalar_type() == at::ScalarType::Int, "selected_experts must be Int32");
    // token_to_scatter_offset stores integer offsets inside each expert segment.
    TORCH_CHECK(token_to_scatter_offset.scalar_type() == at::ScalarType::Int, "token_to_scatter_offset must be Int32");
    // experts_token_count stores one integer routed-row count per expert.
    TORCH_CHECK(experts_token_count.scalar_type() == at::ScalarType::Int, "experts_token_count must be Int32");
    // experts_token_start stores one integer prefix-sum start offset per expert.
    TORCH_CHECK(experts_token_start.scalar_type() == at::ScalarType::Int, "experts_token_start must be Int32");
    // scatter_tokens_offset stores one integer source-token id per scattered row.
    TORCH_CHECK(scatter_tokens_offset.scalar_type() == at::ScalarType::Int, "scatter_tokens_offset must be Int32");

    // Dtype checks (Float32)
    // moe_weights stores float router weights.
    TORCH_CHECK(moe_weights.scalar_type() == at::ScalarType::Float, "moe_weights must be Float32");
    // experts_smooth_scale stores float expert-local smoothing coefficients.
    TORCH_CHECK(experts_smooth_scale.scalar_type() == at::ScalarType::Float, "experts_smooth_scale must be Float32");
    // scatter_per_token_scale stores float dequantization scales for each scattered row.
    TORCH_CHECK(scatter_per_token_scale.scalar_type() == at::ScalarType::Float, "scatter_per_token_scale must be Float32");

    // Logical shape checks. These are especially important here because several
    // output tensors are workspaces produced by the kernel itself, not inputs
    // computed elsewhere.
    TORCH_CHECK(experts_token_count.size(0) == experts_token_start.size(0), "Token count and start tensors must match in size");

    // Block size alignment check for ESIMD vectorized loads
    // Hidden width of each source token row, used to validate the vectorized load contract.
    int64_t hd_size = hidden_states.size(1);
    TORCH_CHECK(hd_size >= 64 && hd_size % 64 == 0,
                "hidden_states inner dimension must be a positive multiple of 64 for XPU block loads, got ", hd_size);

    // Runtime dtype dispatch selects the templated implementation pair.
    // Runtime scalar type of the source hidden-state tensor.
    auto in_dtype = hidden_states.scalar_type();
    // Runtime scalar type requested for the quantized scatter buffer.
    auto out_dtype = scatter_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_scatter_dynamic_quant_impl,
                            selected_experts, moe_weights, token_to_scatter_offset,
                            experts_token_count, experts_token_start, hidden_states,
                            experts_smooth_scale, scatter_tokens, scatter_per_token_scale,
                            scatter_tokens_offset, shared_experts_num);
}

PYBIND11_MODULE(moe_scatter_dynamic_quant_sycl, m) {
    m.def("moe_scatter_dynamic_quant", &moe_scatter_dynamic_quant, "MoE Scatter Dynamic Quant");
}
