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
 * Plain SwiGLU + per-token dynamic quantization.
 *
 * What problem this kernel solves:
 *   - The input row contains two equally sized halves. In transformer-style MLP
 *     blocks this usually corresponds to the two branches used by SwiGLU:
 *       x1 = gate branch
 *       x2 = value branch
 *   - The kernel computes:
 *       swiglu(x1, x2) = (x1 * sigmoid(x1)) * x2
 *   - It then multiplies the result by a precomputed smooth_scale vector.
 *   - Finally, it quantizes that float result independently for each token row.
 *
 * Why "per-token dynamic quantization" matters:
 *   - Every token row can have a different numeric range.
 *   - Using one scale per row preserves more detail than forcing all rows to
 *     share a single global scale.
 *   - The kernel therefore first discovers the max absolute value in a row,
 *     derives a row scale from that max, and only then writes quantized output.
 *
 * Contract:
 *   - hidden_states: [num_tokens, 2 * hidden_size]
 *       Each row is laid out as [x1 | x2].
 *   - smooth_scale:  [hidden_size]
 *       One multiplicative factor per output channel.
 *   - quant_tokens:  [num_tokens, hidden_size]
 *       Quantized result after SwiGLU and smoothing.
 *   - per_token_scale: [num_tokens]
 *       The floating-point scale needed to dequantize each output row later.
 *
 * High-level execution outline:
 *   1. One work-group handles one token row.
 *   2. Threads in that work-group cooperatively process the hidden dimension in
 *      64-element ESIMD blocks.
 *   3. Pass 1 computes the float output, tracks per-thread maxima, and stores
 *      the float intermediates in SLM (shared local memory).
 *   4. The work-group reduces those maxima to one row-wide maximum value.
 *   5. Thread 0 converts that maximum into a per-token quantization scale.
 *   6. Pass 2 rereads the cached float row from SLM, quantizes it, and writes
 *      the packed output tensor.
 *
 * If you are new to SYCL/XPU terminology:
 *   - queue.submit(...) launches work on the device.
 *   - A work-group is a cooperative set of threads that can synchronize.
 *   - SLM (shared local memory) is on-chip scratch memory visible to threads in
 *     the same work-group only. It is much cheaper than going back to global
 *     memory and is used here as a per-row staging buffer.
 *   - ESIMD is Intel's explicit SIMD programming style. The simd<T, N> objects
 *     in this file represent vector registers, not host-side arrays.
 */

// Small type trait used to map an output element type to the largest magnitude
// that type can represent for the quantization scheme used in this file.
template <typename decl_tag> struct QuantMax;
// int8 rows quantize into the symmetric range [-127, 127].
template <> struct QuantMax<int8_t> { static constexpr float value = 127.0f; };
// uint8 here stands for FP8 e4m3fn bytes, whose largest finite value is 448.
template <> struct QuantMax<uint8_t> { static constexpr float value = 448.0f; }; // e4m3fn max value

template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    // This is a software approximation for converting float32 values into the
    // FP8 e4m3fn format. The hardware path is not used here, so the conversion
    // is expressed directly in terms of bit manipulation on each SIMD lane.
    //
    // e4m3fn roughly means:
    //   - 1 sign bit
    //   - 4 exponent bits
    //   - 3 mantissa bits
    //   - "fn" variant semantics used by PyTorch's Float8_e4m3fn dtype
    // Reinterpret each float lane as raw IEEE-754 bits.
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    // Extract the sign bit into the destination position used by e4m3fn.
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    // Clear the sign bit so the rest of the conversion can work on magnitude.
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;
    // Add a rounding bias before truncating mantissa bits.
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;

    // Translate the float32 exponent field into the e4m3fn exponent bias space.
    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    // Keep only the top 3 mantissa bits needed by e4m3fn.
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    // Initialize the output byte lanes to zero before selectively filling them.
    simd<uint8_t, N> res = 0;
    // Normal numbers are those whose exponent still fits the destination range.
    auto is_normal = (exp > 0) & (exp < 16);
    // Overflow means the exponent is too large for e4m3fn.
    auto is_overflow = exp >= 16;
    // Underflow means the value becomes zero in this fast path.
    auto is_underflow = exp <= 0;

    // Normal values keep sign/exponent/mantissa.
    res.merge(sign | (exp << 3) | mantissa, is_normal);
    // Overflow saturates to the largest finite representable e4m3fn value.
    res.merge(sign | 0x7E, is_overflow);
    // Underflow is flushed to signed zero for simplicity and speed.
    res.merge(sign, is_underflow);
    return res;
}

template <typename T_in, typename T_out>
void swiglu_dynamic_quant_impl(
    torch::Tensor& hidden_states,
    torch::Tensor& smooth_scale,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale) {

    // Reuse PyTorch's currently active XPU stream so this extension obeys the
    // same ordering semantics as surrounding PyTorch ops.
    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    // Raw device pointers are taken once on the host side, then captured into
    // the kernel lambda. Inside the device kernel there is no Tensor API, only
    // pointer arithmetic on these contiguous buffers.
    // Base pointer for the input matrix containing [x1 | x2] rows.
    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    // Base pointer for the 1D smoothing vector applied after SwiGLU.
    auto smooth_scale_ptr = smooth_scale.data_ptr<float>();
    // Base pointer for the quantized output matrix.
    auto quant_tokens_ptr = reinterpret_cast<T_out*>(quant_tokens.data_ptr());
    // Base pointer for the per-token dequantization scale output.
    auto per_token_scale_ptr = per_token_scale.data_ptr<float>();

    // Number of token rows in the batch.
    int num_tokens = hidden_states.size(0);
    // Logical hidden size after splitting the concatenated [x1 | x2] row in half.
    int hidden_size = hidden_states.size(1) / 2;
    // Maximum magnitude representable in the destination type.
    constexpr float quant_max = QuantMax<T_out>::value;

    // The launch helper is templated through integral_constant tags so the
    // compiler sees UNROLL and SLM_BYTES as compile-time constants. That lets
    // the ESIMD compiler fully specialize load/store patterns for each variant.
    auto launch_swiglu = [&](auto unroll_tag, auto slm_tag) {
        // Number of 64-lane chunks processed per inner-loop iteration.
        constexpr int UNROLL = decltype(unroll_tag)::value;
        // Number of SLM bytes reserved for this specific kernel variant.
        constexpr uint32_t SLM_BYTES = decltype(slm_tag)::value;
        // ESIMD vector width used by the block load/store path.
        constexpr int CHUNK = 64;
        // Total number of hidden elements covered by one unrolled iteration.
        constexpr int BS = CHUNK * UNROLL;

        // hidden_size is guaranteed to be divisible by 64. The hidden row is
        // divided into BS-sized blocks, and each thread visits a strided subset
        // of those blocks.
        // Number of BS-sized blocks needed to cover one token row.
        int num_blocks = hidden_size / BS;
        // Number of cooperating threads per work-group, capped by hardware-friendly 64.
        int wg_size = std::min(num_blocks, 64);

        // 2D launch shape:
        //   dimension 0 -> token rows
        //   dimension 1 -> threads cooperating on one row
        // Global launch range: one row dimension and one intra-row thread dimension.
        sycl::range<2> GlobalRange(num_tokens, wg_size);
        // Local launch range: one work-group per row and wg_size threads inside it.
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                // Row index handled by this work-group/thread combination.
                int token_idx = item.get_global_id(0);
                // Thread id inside the work-group that collaborates on one row.
                int tid = item.get_local_id(1);

                // Every work-group gets its own SLM allocation. We use it for:
                //   1. the float intermediate row
                //   2. a small array of per-thread maxima for reduction
                //   3. one stored inverse scale value broadcast to all threads
                slm_init(SLM_BYTES);

                // Pointer to the first element of the current input row.
                auto in_row_ptr = hidden_states_ptr + token_idx * (hidden_size * 2);
                // Pointer to the first element of the current output row.
                auto out_row_ptr = quant_tokens_ptr + token_idx * hidden_size;
                // The first hidden_size * sizeof(float) bytes are used to cache
                // the float SwiGLU output row. The reduction scratch region is
                // placed immediately after that cache.
                uint32_t reduction_base = hidden_size * sizeof(float); // Safely offset reduction array

                // Per-thread running maximum over the CHUNK lanes processed by this thread.
                simd<float, CHUNK> thread_max_vec = 0.0f;

                // Pass 1: read source tensors from global memory, compute the
                // float SwiGLU output, accumulate the max absolute value seen by
                // this thread, and store the float result into SLM.
                //
                // This two-pass structure avoids recomputing SwiGLU during
                // quantization while also avoiding an extra writeback to global
                // memory for the intermediate float row.
                // Walk the row in a thread-strided pattern so the work-group covers the row cooperatively.
                for (int bid = tid; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    // Unroll within each assigned block to increase vector throughput.
                    for (int u = 0; u < UNROLL; ++u) {
                        // Load the gate half for this vector segment.
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(
                            in_row_ptr + bid * BS + u * CHUNK);
                        // Load the value half for the matching vector segment.
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(
                            in_row_ptr + hidden_size + bid * BS + u * CHUNK);
                        // Load the smooth-scale coefficients for this output slice.
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + bid * BS + u * CHUNK);
                        
                        // Compute sigmoid(x1) lane-wise, then the SwiGLU output
                        // (x1 * sigmoid(x1)) * x2, and finally apply the smooth
                        // scale that was provided by the caller.
                        // Compute sigmoid of the gate branch.
                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        // Combine gate, value, and smooth-scale into the final float activation.
                        simd<float, CHUNK> scaled = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        
                        // Dynamic quantization needs the largest absolute value
                        // in the full row so that every later quantized value
                        // fits into the target numeric range.
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled));
                        
                        // SLM block stores move the float intermediate into the
                        // work-group scratchpad. The 4 x 16-lane layout is used
                        // because the block-store primitive operates naturally on
                        // these 16-float chunks.
                        // Byte offset into the SLM cache region for this vector segment.
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
#pragma unroll
                        // Store the 64-lane vector as four 16-lane blocks because that matches the SLM primitive.
                        for (int i = 0; i < 4; ++i) {
                            slm_block_store<float, 16>(base_offset + i * 64, scaled.template select<16, 1>(i * 16));
                        }
                    }
                }

                // Reduce this thread's SIMD register down to a single scalar.
                float local_max = hmax<float, float, CHUNK>(thread_max_vec);

                // Each thread publishes one scalar max into SLM, then the group
                // synchronizes so thread 0 can complete the final reduction.
                slm_block_store<float, 4>(reduction_base + tid * 16, simd<float, 4>(local_max));
                barrier();

                // Default inverse scale used until thread 0 publishes the real one.
                float inv_scale = 1.0f;
                // WG-level reduction for the global row max.
                // Only one thread performs this small serial loop because the
                // number of entries is bounded by wg_size <= 64.
                if (tid == 0) {
                    // Row-wise maximum magnitude across all threads in the group.
                    float token_max = 0.0f;
                    // Read back each thread's contribution from SLM.
                    for (int i = 0; i < wg_size; i++) {
                        // Only the first lane is meaningful because each store wrote one scalar value.
                        float thread_m = slm_block_load<float, 4>(reduction_base + i * 16)[0];
                        if (thread_m > token_max) token_max = thread_m;
                    }
                    // quant_max is the largest representable magnitude in the
                    // destination format. Dividing by it yields the dequant
                    // scale for this row.
                    float token_scale_val = token_max / quant_max;
                    // All-zero rows would otherwise produce a zero divisor.
                    token_scale_val = (token_scale_val == 0.0f) ? 1.0f : token_scale_val;
                    per_token_scale_ptr[token_idx] = token_scale_val;
                    // The inverse is what the quantization pass needs, so store
                    // it once in SLM for all threads to reuse.
                    slm_block_store<float, 4>(reduction_base + 1024, simd<float, 4>(1.0f / token_scale_val)); // inverse scale
                }
                barrier();
                // Every thread reloads the shared inverse scale before the quantization pass.
                inv_scale = slm_block_load<float, 4>(reduction_base + 1024)[0];

                // Pass 2: reload the cached float row from SLM, multiply by the
                // inverse row scale, convert to the output dtype, and write the
                // final quantized row to global memory.
                // Visit the same row slices again, this time reading from SLM instead of recomputing.
                for (int bid = tid; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    // Reuse the same unrolled traversal shape as pass 1.
                    for (int u = 0; u < UNROLL; ++u) {
                        // Byte offset of the cached float segment in SLM.
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
                        // Temporary register holding the cached float activations.
                        simd<float, CHUNK> cached_tokens;

#pragma unroll
                        // Reassemble the 64-lane vector from four 16-lane SLM loads.
                        for (int i = 0; i < 4; ++i) {
                            cached_tokens.template select<16, 1>(i * 16) = slm_block_load<float, 16>(base_offset + i * 64);
                        }

                        // Register that will hold the converted quantized bytes or ints.
                        simd<T_out, CHUNK> quantized_out;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            // rnde performs round-to-nearest-even conversion.
                            quantized_out = rnde<float>(cached_tokens * inv_scale);
                        } else { // FP8 e4m3fn
                            quantized_out = fast_cvt_float_to_e4m3fn<CHUNK>(cached_tokens * inv_scale);
                        }

                        // Commit the quantized vector back to global memory at the correct row offset.
                        block_store<T_out, CHUNK>(out_row_ptr + bid * BS + u * CHUNK, quantized_out);
                    }
                }
            });
        });
    };

    // SLM size must be a compile-time constant for the specialized kernel type.
    // Each bucket reserves enough bytes for:
    //   - hidden_size float intermediates
    //   - a small fixed-size reduction area
    // The comments on the right show the approximate layout math.
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

    // Minimum number of blocks we want the work-group to have available.
    int target_wg = 1;
    // Number of 64-element chunks in one row.
    int num_chunks = hidden_size / 64;
    // Fallback unroll choice if no wider clean divisor is found.
    int best_unroll = 1;

    // Prefer the largest clean unroll factor that still leaves enough blocks to
    // keep the work-group busy. Larger unroll usually means fewer loop/control
    // overheads, but only if the row is large enough to supply enough work.
    for (int u : {8, 4, 2}) {
        // Accept the first widest divisor that still leaves enough per-row work items.
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

// Runtime dtype dispatch macro bridging PyTorch scalar types to C++ templates.
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

    // DeviceGuard ensures that all Tensor operations and the launched SYCL
    // kernel are associated with the input tensor's device, even if the caller
    // currently has another device selected in PyTorch.
    at::DeviceGuard guard(hidden_states.device());

    // Require contiguous input rows so the block loads map to simple pointer arithmetic.
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    // Require the smooth-scale vector to be contiguous for vectorized reads.
    TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
    // Require the output buffer to be contiguous for vectorized stores.
    TORCH_CHECK(quant_tokens.is_contiguous(), "quant_tokens must be contiguous");

    // The outer function is a thin validation and dispatch layer. The actual
    // algorithm lives in swiglu_dynamic_quant_impl<T_in, T_out>.
    // Runtime scalar type of the input tensor.
    auto in_dtype = hidden_states.scalar_type();
    // Runtime scalar type of the destination tensor.
    auto out_dtype = quant_tokens.scalar_type();

    DISPATCH_QUANT_IMPL(swiglu_dynamic_quant_impl,
                        hidden_states, smooth_scale, quant_tokens, per_token_scale);
}

PYBIND11_MODULE(swiglu_dynamic_quant_sycl, m) {
    m.def("swiglu_dynamic_quant", &swiglu_dynamic_quant, "SwiGLU Dynamic Quant");
}