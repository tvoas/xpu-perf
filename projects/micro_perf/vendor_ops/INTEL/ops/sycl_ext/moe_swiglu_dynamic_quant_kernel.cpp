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
 * Grouped MoE SwiGLU + per-token dynamic quantization.
 *
 * Where this sits in an MoE pipeline:
 *   - A routing stage has already decided which expert owns each token.
 *   - Tokens have already been "scattered" into expert-grouped order.
 *   - This kernel does not decide routing; it consumes routed data.
 *   - For each scattered row, it applies the expert-local SwiGLU path and then
 *     dynamically quantizes the result.
 *
 * Contract:
 *   - scatter_tokens is already laid out in dispatch order as
 *     [dispatch_tokens, 2 * hidden_size].
 *     Each row is [x1 | x2] for the expert that owns that row.
 *   - experts_token_count / experts_token_start describe contiguous expert
 *     segments inside that dispatch buffer. They are mainly metadata for the
 *     surrounding pipeline and validation.
 *   - scatter_expert_ids gives the owning expert for each dispatch row.
 *     This kernel uses that array to choose the correct smooth_scale row.
 *
 * Execution model:
 *   - One work-group handles one scattered token row.
 *   - Threads in that work-group cooperatively process the hidden dimension.
 *   - Like the plain SwiGLU kernel, the implementation is split into two passes:
 *       1. compute float output + find row max + cache in SLM
 *       2. quantize cached values using the derived row scale
 *
 * Key difference from the plain SwiGLU kernel:
 *   - smooth_scale is no longer a single vector shared by every row.
 *   - Each row belongs to an expert, and that expert has its own smooth_scale.
 *   - scatter_expert_ids[flat_idx] tells the kernel which expert row to read.
 */

// Small type trait that maps an output element type to its quantization ceiling.
template <typename decl_tag> struct QuantMax;
// int8 rows use 127 as the positive saturation point.
template <> struct QuantMax<int8_t> { static constexpr float value = 127.0f; };
// FP8 e4m3fn rows use 448 as the largest finite representable value.
template <> struct QuantMax<uint8_t> { static constexpr float value = 448.0f; }; // e4m3fn max value

// Vectorized software emulation for float32 -> float8_e4m3fn conversion
template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    // Same software FP8 conversion strategy as in the plain kernel. It is kept
    // local here so this file can compile independently as a standalone module.
    // Raw float bits for each SIMD lane.
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    // Destination sign bit.
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    // Magnitude bits with sign removed.
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;

    // Rounding tie-to-even approximation (add half of the 20-bit shifted fractional part)
    // Rounding bias before truncating mantissa precision.
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;

    // Rebiased exponent in e4m3fn space.
    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    // Top 3 mantissa bits kept for the reduced-precision format.
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    // Zero-initialized destination vector before mask merges.
    simd<uint8_t, N> res = 0;

    // Mask for values representable without saturation.
    auto is_normal = (exp > 0) & (exp < 16);
    // Mask for values that must be clamped high.
    auto is_overflow = exp >= 16;
    // Mask for values that become zero in this simplified path.
    auto is_underflow = exp <= 0;

    res.merge(sign | (exp << 3) | mantissa, is_normal);
    res.merge(sign | 0x7E, is_overflow); // 0x7E is 448.0 (Maximum e4m3fn value)
    res.merge(sign, is_underflow);       // Fast fallback: flush subnormals to 0

    return res;
}

template <typename T_in, typename T_out>
void moe_swiglu_dynamic_quant_impl(
    torch::Tensor& scatter_tokens,
    torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& scatter_expert_ids,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale,
    int64_t total_experts_num,
    int64_t max_token_num) {

    // A cheap host-side guard avoids launching an empty kernel when the routed
    // batch contains no work.
    if (max_token_num <= 0 || total_experts_num <= 0) {
        return;
    }

    // Use PyTorch's current XPU stream for ordering with surrounding ops.
    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    // Raw pointers captured by the device lambda. Once execution enters the
    // kernel, all tensor semantics have been lowered to address arithmetic.
    // Base pointer for the scattered [x1 | x2] activation rows.
    auto scatter_tokens_ptr = reinterpret_cast<T_in*>(scatter_tokens.data_ptr());
    // Base pointer for the expert-specific smoothing matrix.
    auto smooth_scale_ptr = smooth_scale.data_ptr<float>();
    // Base pointer for the expert-id lookup table, one id per scattered row.
    auto scatter_expert_ids_ptr = scatter_expert_ids.data_ptr<int32_t>();
    // Base pointer for the quantized output rows.
    auto quant_tokens_ptr = reinterpret_cast<T_out*>(quant_tokens.data_ptr());
    // Base pointer for the row-wise dequantization scale output.
    auto per_token_scale_ptr = per_token_scale.data_ptr<float>();

    // Logical hidden width after splitting [x1 | x2] rows in half.
    int hidden_size = scatter_tokens.size(1) / 2;
    // Number of rows already routed into dispatch order.
    int num_scattered = scatter_tokens.size(0);
    // Positive saturation point for the destination quantized dtype.
    constexpr float quant_max = QuantMax<T_out>::value;

    // Compile-time launch specialization. UNROLL influences how many 64-wide
    // vector chunks each inner loop iteration processes. SLM_BYTES fixes the
    // local memory reservation for this specific kernel instantiation.
    auto launch_swiglu = [&](auto unroll_tag, auto slm_tag) {
        // Number of vector chunks processed per inner-loop iteration.
        constexpr int UNROLL = decltype(unroll_tag)::value;
        // Local-memory footprint reserved for this kernel variant.
        constexpr uint32_t SLM_BYTES = decltype(slm_tag)::value;
        // ESIMD vector width used by this kernel family.
        constexpr int CHUNK = 64;
        // Hidden elements covered by one unrolled iteration.
        constexpr int BS = CHUNK * UNROLL;

        // Number of BS-sized blocks needed to cover one scattered row.
        int num_blocks = hidden_size / BS;
        // Cooperative thread count for one row, capped at 64.
        int wg_size = std::min(num_blocks, 64);

        // Each group in dimension 0 corresponds to exactly one dispatch row.
        // Dimension 1 is the cooperative thread dimension within that row.
        sycl::range<2> GlobalRange(num_scattered, wg_size);
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                // SLM plays the same role as in the non-MoE kernel: it holds
                // the float intermediate row and the temporary reduction state.
                slm_init(SLM_BYTES);

                // Thread id within the current row's work-group.
                const int loc_id = item.get_local_id(1);
                // Flattened scattered-row index owned by this work-group.
                const int flat_idx = item.get_group(0);

                // The scatter stage wrote one expert id per dispatch row. That
                // id selects which smooth_scale row is used for this token.
                // Expert id associated with this scattered row.
                const int expert_idx = scatter_expert_ids_ptr[flat_idx];
                if (expert_idx < 0 || expert_idx >= total_experts_num) {
                    return;
                }

                // Each scattered row stores two hidden vectors back-to-back:
                // the gate half and the value half used by SwiGLU.
                // Pointer to the first element of this row's concatenated input.
                T_in* scatter_token_base = scatter_tokens_ptr + flat_idx * 2 * hidden_size;
                // Pointer to the first element of this row's output buffer.
                T_out* output_base = quant_tokens_ptr + flat_idx * hidden_size;
                // The cached float row occupies the first hidden_size floats in
                // SLM. Reduction scratch space starts immediately after that.
                uint32_t reduction_base = hidden_size * sizeof(float); // Safely offset reduction array

                // Running per-thread maximum over all vector chunks assigned to this thread.
                simd<float, CHUNK> thread_max_vec = 0.0f;

                // Pass 1: compute expert-local SwiGLU in float, track the max
                // absolute value needed for quantization, and stage the float
                // row in SLM so pass 2 does not need to recompute it.
                // Thread-strided loop over the blocks that make up one scattered row.
                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    // Unroll within each block to reduce loop overhead.
                    for (int u = 0; u < UNROLL; ++u) {
                        // Load the gate branch slice for this row.
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(
                            scatter_token_base + bid * BS + u * CHUNK);
                        // Load the value branch slice for the same logical output positions.
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(
                            scatter_token_base + hidden_size + bid * BS + u * CHUNK);
                        // Load the expert-local smooth-scale slice selected by expert_idx.
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_idx * hidden_size + bid * BS + u * CHUNK);

                        // expert_idx selects the smooth scale row that belongs
                        // to the expert owning this scattered token.
                        // Compute the gate activation for this vector slice.
                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        // Form the final float output slice after SwiGLU and expert smoothing.
                        simd<float, CHUNK> scaled_swiglu_tokens = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;

                        // The per-row dynamic scale is driven by the largest
                        // absolute value produced anywhere in this row.
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled_swiglu_tokens));

                        // Byte offset into the SLM cache region for this float slice.
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
#pragma unroll
                        // Emit the 64-lane float slice as four SLM block stores.
                        for (int i = 0; i < 4; ++i) {
                            slm_block_store<float, 16>(base_offset + i * 64, scaled_swiglu_tokens.template select<16, 1>(i * 16));
                        }
                    }
                }

                // Collapse this thread's vector max down to one scalar.
                float thread_max = hmax<float, float, CHUNK>(thread_max_vec);

                // Publish one scalar per thread, then synchronize before the
                // single-thread reduction.
                slm_block_store<float, 4>(reduction_base + loc_id * 16, simd<float, 4>(thread_max));
                barrier();

                // Row scale defaults to 1 until thread 0 computes the true value.
                float this_token_scale = 1.0f;

                if (loc_id == 0) {
                    // Serial reduction across the thread-local maxima. This is
                    // cheap because the work-group width is at most 64.
                    // Accumulator for the row-wise max across all threads in the work-group.
                    float max_value_final = 0.0f;
                    // Scan the per-thread maxima that were published into SLM.
                    for (int i = 0; i < wg_size; i++) {
                        // Only val[0] carries data because each thread stored one scalar max.
                        simd<float, 4> val = slm_block_load<float, 4>(reduction_base + i * 16);
                        if (val[0] > max_value_final) max_value_final = val[0];
                    }

                    // Store the dequant scale, not its reciprocal. The caller
                    // can later reconstruct approximate float values as:
                    //   dequantized ~= quantized * this_token_scale
                    float raw_token_scale = max_value_final / quant_max;
                    this_token_scale = raw_token_scale == 0.0f ? 1.0f : raw_token_scale;

                    // Broadcast the scale through SLM so every thread uses the
                    // same value in pass 2.
                    slm_block_store<float, 4>(reduction_base + 1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                // Reload the broadcast row scale from SLM.
                this_token_scale = slm_block_load<float, 4>(reduction_base + 1024)[0];
                // Precompute the reciprocal because quantization multiplies by it repeatedly.
                float recip_scale = 1.0f / this_token_scale;

                // Pass 2: reload the cached float row, multiply by the inverse
                // of the row scale, convert to the output type, and store the
                // quantized result.
                // Walk the row again using the same cooperative partitioning.
                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    // Reuse the same unroll factor for pass 2.
                    for (int u = 0; u < UNROLL; ++u) {
                        // Byte offset of the cached float slice within SLM.
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
                        // Register that will hold the reconstructed cached float slice.
                        simd<float, CHUNK> cached_tokens;

#pragma unroll
                        // Rebuild the 64-lane vector from four 16-lane SLM loads.
                        for (int i = 0; i < 4; ++i) {
                            cached_tokens.template select<16, 1>(i * 16) = slm_block_load<float, 16>(base_offset + i * 64);
                        }

                        // Output register holding quantized lanes before the final store.
                        simd<T_out, CHUNK> quantized_out;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized_out = rnde<float>(cached_tokens * recip_scale);
                        } else {
                            quantized_out = fast_cvt_float_to_e4m3fn<CHUNK>(cached_tokens * recip_scale);
                        }

                        // Write the quantized slice into this row's output buffer.
                        block_store<T_out, CHUNK>(output_base + bid * BS + u * CHUNK, quantized_out);
                    }
                }

                if (loc_id == 0) {
                    // One scale value is stored per scattered row. The caller's
                    // bookkeeping tensors preserve the mapping back to experts.
                    block_store<float, 1>(per_token_scale_ptr + flat_idx, this_token_scale);
                }
            });
        });
    };

    // Bucket hidden sizes so each specialized kernel instance has a fixed local
    // memory footprint known at compile time. This is a common ESIMD pattern
    // when the temporary storage depends on the row width.
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

    // Minimum amount of per-row parallel work we want after unrolling.
    int target_wg = 2;

    // Number of CHUNK-sized vector slices in one row.
    int num_chunks = hidden_size / 64;
    // Conservative fallback unroll value.
    int best_unroll = 1;

    // Choose the widest clean unroll that still leaves enough independent work
    // items across the row. This is a simple heuristic balancing vectorization
    // against occupancy.
    for (int u : {8, 4, 2}) {
        // Keep the first widest divisor that still leaves enough blocks for the work-group.
        if (num_chunks % u == 0 && (num_chunks / u) >= target_wg) {
            best_unroll = u;
            break;
        }
    }

    switch (best_unroll) {
        case 8:  return dispatch_slm(std::integral_constant<int, 8>{});
        case 4:  return dispatch_slm(std::integral_constant<int, 4>{});
        case 2:  return dispatch_slm(std::integral_constant<int, 2>{});
        default: return dispatch_slm(std::integral_constant<int, 1>{});
    }
}

// Outer dispatch macros to select implementation
// Macro translating runtime dtypes into the concrete template instantiation.
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

void moe_swiglu_dynamic_quant(
    torch::Tensor& scatter_tokens,
    torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& scatter_expert_ids,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale,
    int64_t total_experts_num,
    int64_t max_token_num) {

    // Guard against the caller having another XPU device selected. All checks
    // and the launched kernel should be interpreted relative to scatter_tokens.
    at::DeviceGuard guard(scatter_tokens.device());

    // Contiguity checks
    // The scattered input rows must be contiguous for block loads to work directly.
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    // The expert smooth-scale matrix must be contiguous for vectorized reads.
    TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
    // The per-expert count metadata must be laid out densely.
    TORCH_CHECK(experts_token_count.is_contiguous(), "experts_token_count must be contiguous");
    // The per-expert start offsets must be laid out densely.
    TORCH_CHECK(experts_token_start.is_contiguous(), "experts_token_start must be contiguous");
    // The per-row expert-id table must be contiguous because each work-group indexes it directly.
    TORCH_CHECK(scatter_expert_ids.is_contiguous(), "scatter_expert_ids must be contiguous");
    // The quantized output matrix must be contiguous for block stores.
    TORCH_CHECK(quant_tokens.is_contiguous(), "quant_tokens must be contiguous");
    // The per-row scale output vector must be contiguous for scalar writes.
    TORCH_CHECK(per_token_scale.is_contiguous(), "per_token_scale must be contiguous");

    // Dtype checks
    TORCH_CHECK(smooth_scale.scalar_type() == at::ScalarType::Float, "smooth_scale must be Float32");
    TORCH_CHECK(per_token_scale.scalar_type() == at::ScalarType::Float, "per_token_scale must be Float32");
    TORCH_CHECK(experts_token_count.scalar_type() == at::ScalarType::Int, "experts_token_count must be Int32");
    TORCH_CHECK(experts_token_start.scalar_type() == at::ScalarType::Int, "experts_token_start must be Int32");
    TORCH_CHECK(scatter_expert_ids.scalar_type() == at::ScalarType::Int, "scatter_expert_ids must be Int32");

    // Shape checks. These establish the exact contract between the Python
    // wrapper and the device kernel before any raw pointer arithmetic happens.
    // Number of scattered rows produced by the routing stage.
    int64_t num_scattered = scatter_tokens.size(0);
    // Physical row width, which should equal 2 * hidden_size.
    int64_t hidden_size2 = scatter_tokens.size(1);
    TORCH_CHECK(hidden_size2 % 2 == 0, "scatter_tokens hidden dimension must be divisible by 2");
    // Logical output hidden width after separating gate and value halves.
    int64_t hidden_size = hidden_size2 / 2;

    // Block size alignment check for ESIMD vectorized loads
    TORCH_CHECK(hidden_size >= 64 && hidden_size % 64 == 0,
                "hidden_size must be a positive multiple of 64 for vectorized XPU block loads, got ", hidden_size);
    TORCH_CHECK(hidden_size <= 32256,
                "hidden_size exceeds current SLM-backed MoeSwigluDynamicQuant limit of 32256, got ", hidden_size);

    TORCH_CHECK(quant_tokens.size(0) == num_scattered && quant_tokens.size(1) == hidden_size,
                "quant_tokens shape mismatch");
    TORCH_CHECK(smooth_scale.size(0) == total_experts_num && smooth_scale.size(1) == hidden_size,
                "smooth_scale shape mismatch");
    TORCH_CHECK(experts_token_count.size(0) == total_experts_num, "experts_token_count size mismatch");
    TORCH_CHECK(experts_token_start.size(0) == total_experts_num, "experts_token_start size mismatch");
    TORCH_CHECK(scatter_expert_ids.size(0) == num_scattered, "scatter_expert_ids size mismatch");
    TORCH_CHECK(per_token_scale.size(0) == num_scattered, "per_token_scale size mismatch");

    // Runtime dtype dispatch chooses the specialized templated kernel.
    // Runtime scalar type of the scattered input rows.
    auto in_dtype = scatter_tokens.scalar_type();
    // Runtime scalar type requested for the quantized output rows.
    auto out_dtype = quant_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_swiglu_dynamic_quant_impl,
                            scatter_tokens, smooth_scale, experts_token_count,
                            experts_token_start, scatter_expert_ids, quant_tokens, per_token_scale,
                            total_experts_num, max_token_num);
}

PYBIND11_MODULE(moe_swiglu_dynamic_quant_sycl, m) {
    m.def("moe_swiglu_dynamic_quant", &moe_swiglu_dynamic_quant, "MoE SwiGLU Dynamic Quant");
}
