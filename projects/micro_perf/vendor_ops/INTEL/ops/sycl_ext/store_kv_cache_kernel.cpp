// store_kv_cache_kernel.cpp — v6: streaming stores for INT8 quantized writes
//
// v6 changes vs v5:
//   - Streaming store hints on INT8 cache writes (L1+L2 bypass)
//   - KV cache is write-once: streaming avoids read-for-ownership and L2 pollution
//   - INT8 benefits most due to scale reads competing for L2 with cache writes
//   - BF16/FP8 paths remain unchanged (no measurable benefit from streaming)
//   - Uses sycl::ext::intel::experimental cache_control with annotated_ptr
//
// v5 changes vs v4:
//   - Accept packed_qkv directly instead of pre-sliced key/value tensors
//   - Kernel reads K/V via head offsets into packed_qkv — zero Python-side copy
//   - Source stride = total_heads × head_dim (skips Q heads between tokens)
//
// Optimizations:
//   1. nd_range: one work-group per token, kv_head sub-groups per group
//   2. vec8 loads/stores: 8 bf16 = 16 bytes per transaction
//   3. Sub-group size 16: 16 threads × 8 elements = 128 = head_dim in one pass
//   4. __restrict__ + pre-computed strides
//   5. reqd_sub_group_size attribute for compiler to use native SIMD width
//   6. No slice/contiguous in Python — kernel reads packed_qkv with stride
//   7. Streaming stores: cache bypass for write-once KV cache data

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

// --- Helper: bf16 (uint16_t) -> fp32 via bit shift ---
static inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

// --- Helper: fp32 -> FP8 E4M3 (uint8_t) ---
// FP8 E4M3FN: 1 sign + 4 exponent + 3 mantissa, bias=7, max=448.0, no inf
static inline uint8_t fp32_to_fp8_e4m3(float v) {
    if (sycl::isnan(v)) {
        return 0x7F;
    }

    // Clamp to representable range
    v = sycl::fmin(sycl::fmax(v, -448.0f), 448.0f);
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));

    uint32_t sign = (bits >> 24) & 0x80;  // sign bit → bit7 of fp8
    uint32_t exp = (bits >> 23) & 0xFF;   // fp32 exponent (biased 127)
    uint32_t mant = bits & 0x7FFFFF;      // fp32 mantissa (23 bits)

    if (exp == 0) {
        // fp32 zero or subnormal → fp8 zero
        return static_cast<uint8_t>(sign);
    }

    // Re-bias exponent: fp32 bias=127, fp8_e4m3 bias=7
    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 7;

    if (new_exp > 15) {
        // Overflow → max representable finite value (0x7E = 448.0), NOT inf (E4M3FN has no inf)
        return static_cast<uint8_t>(sign | 0x7E);
    }

    if (new_exp <= 0) {
        // Subnormal in fp8: mantissa * 2^-9.
        const int right_shift = 21 - new_exp;
        if (right_shift > 31) {
            return static_cast<uint8_t>(sign);  // too small → zero
        }

        uint32_t full_mant = (1 << 23) | mant;
        uint32_t fp8_mant = full_mant >> right_shift;
        uint32_t round_bit = (full_mant >> (right_shift - 1)) & 1;
        uint32_t sticky = (full_mant & ((1u << (right_shift - 1)) - 1)) != 0 ? 1 : 0;
        fp8_mant += (round_bit && (sticky || (fp8_mant & 1)));

        if (fp8_mant >= 8) {
            return static_cast<uint8_t>(sign | 0x08);
        }
        return static_cast<uint8_t>(sign | fp8_mant);
    }

    // Normal case: truncate 23-bit mantissa to 3-bit with round-to-nearest-even
    uint32_t fp8_mant = mant >> 20;  // top 3 bits of mantissa
    uint32_t round_bit = (mant >> 19) & 1;
    uint32_t sticky = (mant & 0x7FFFF) != 0 ? 1 : 0;
    fp8_mant += (round_bit && (sticky || (fp8_mant & 1)));

    if (fp8_mant > 7) {
        fp8_mant = 0;
        new_exp += 1;
        if (new_exp > 15) {
            return static_cast<uint8_t>(sign | 0x7E);  // overflow after rounding
        }
    }

    if (new_exp == 15 && fp8_mant > 6) {
        return static_cast<uint8_t>(sign | 0x7E);
    }

    return static_cast<uint8_t>(sign | (new_exp << 3) | fp8_mant);
}

// --- Helper: FP8 E4M3 (uint8_t) -> fp32 ---
static inline float fp8_e4m3_to_fp32(uint8_t v) {
    uint32_t sign = (static_cast<uint32_t>(v) & 0x80) << 24;     // bit31
    uint32_t exp = (static_cast<uint32_t>(v) >> 3) & 0x0F;       // 4-bit exponent
    uint32_t mant = static_cast<uint32_t>(v) & 0x07;              // 3-bit mantissa

    if (exp == 0 && mant == 0) {
        // Zero
        float f;
        uint32_t bits = sign;
        __builtin_memcpy(&f, &bits, sizeof(float));
        return f;
    }

    if (exp == 0) {
        // Subnormal: value = (-1)^s × 0.mant × 2^(-6)
        // Convert to fp32: normalize
        float val = static_cast<float>(mant) / 8.0f;  // 0.mant in [0.125, 0.875]
        val *= (1.0f / 64.0f);  // × 2^(-6)
        return (v & 0x80) ? -val : val;
    }

    // Normal: rebias exponent from fp8 bias=7 to fp32 bias=127
    uint32_t fp32_exp = exp - 7 + 127;
    uint32_t fp32_mant = mant << 20;  // 3-bit → 23-bit mantissa
    uint32_t bits = sign | (fp32_exp << 23) | fp32_mant;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

// --- Vec types for vectorized memory access ---
struct alignas(16) bf16x8_t { uint16_t d[8]; };
struct alignas(8)  int8x8_t { int8_t d[8]; };
struct alignas(8)  uint8x8_t { uint8_t d[8]; };
struct alignas(8)  bf16x4_t { uint16_t x, y, z, w; };
struct alignas(4)  int8x4_t { int8_t x, y, z, w; };

// Sub-group size for Intel Xe2 (BMG/B60): 16 threads
// 16 threads × 8 elements/thread = 128 = head_dim
constexpr int SUBGROUP_SIZE = 16;

struct Bf16CopyConverter {
    static inline uint16_t convert(uint16_t v, const float*) {
        return v;
    }
};

struct Int8Converter {
    static inline int8_t convert(uint16_t v, const float* scale) {
        float f = bf16_to_fp32(v) * (*scale);
        f = sycl::clamp(sycl::rint(f), -127.0f, 127.0f);
        return static_cast<int8_t>(f);
    }
};

struct Fp8Converter {
    static inline uint8_t convert(uint16_t v, const float* scale) {
        return fp32_to_fp8_e4m3(bf16_to_fp32(v) * (*scale));
    }
};

// --- Streaming store: bypass L1+L2 caches for write-once cache data ---
template<typename T>
inline void streaming_store(T* ptr, const T& val) {
    namespace cc = sycl::ext::intel::experimental;
    namespace ep = sycl::ext::oneapi::experimental;
    ep::annotated_ptr<T, decltype(ep::properties{
        cc::write_hint<cc::cache_control<cc::cache_mode::streaming,
                                          cc::cache_level::L1, cc::cache_level::L2>>
    })> ap(ptr);
    *ap = val;
}

template <typename CacheT, typename VecT, typename Converter, bool StreamingWrite = false>
void store_kv_cache_single_linear_impl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    const float* __restrict__ scale_ptr,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int max_kv_len = static_cast<int>(cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int src_token_stride = total_heads * head_dim;
    const int src_offset = static_cast<int>(src_head_start) * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    CacheT* __restrict__ cache_ptr = reinterpret_cast<CacheT*>(cache.data_ptr());

    // P0: 1 WG per (token, head) — reduces write scatter, improves coalescing
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;
            const int src_base = token_idx * src_token_stride + src_offset + head_idx * head_dim;
            const int dst_base = batch_idx * cache_batch_stride
                               + head_idx * cache_head_stride
                               + (cache_start_i + local_q_pos) * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;
                    const int src_off = src_base + dim_off;
                    const int dst_off = dst_base + dim_off;

                    bf16x8_t src8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + src_off);
                    const float* __restrict__ scale = scale_ptr == nullptr
                        ? nullptr
                        : scale_ptr + head_idx * head_dim + dim_off;

                    VecT out;
                    #pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        out.d[j] = Converter::convert(src8.d[j], scale == nullptr ? nullptr : scale + j);
                    }
                    if constexpr (StreamingWrite) {
                        streaming_store(reinterpret_cast<VecT*>(cache_ptr + dst_off), out);
                    } else {
                        *reinterpret_cast<VecT*>(cache_ptr + dst_off) = out;
                    }
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    const int src_off = src_base + i;
                    const int dst_off = dst_base + i;
                    const float* __restrict__ scale = scale_ptr == nullptr
                        ? nullptr
                        : scale_ptr + head_idx * head_dim + i;

                    auto conv_val = Converter::convert(qkv_ptr[src_off], scale);
                    if constexpr (StreamingWrite) {
                        streaming_store(cache_ptr + dst_off, conv_val);
                    } else {
                        cache_ptr[dst_off] = conv_val;
                    }
                }
            }
        });
    });
}

template <typename CacheT, typename VecT, typename Converter, bool StreamingWrite = false>
void store_kv_cache_single_paged_impl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    const float* __restrict__ scale_ptr,
    torch::Tensor block_table,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int block_size_i = static_cast<int>(offset_major ? cache.size(1) : cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);
    const int num_blocks_per_seq = static_cast<int>(block_table.size(1));

    const int src_token_stride = total_heads * head_dim;
    const int src_offset = static_cast<int>(src_head_start) * head_dim;
    const int head_size_stride = block_size_i * head_dim;
    const int block_stride = kv_head * head_size_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    CacheT* __restrict__ cache_ptr = reinterpret_cast<CacheT*>(cache.data_ptr());
    const int32_t* __restrict__ bt_ptr = block_table.data_ptr<int32_t>();

    // P0: 1 WG per (token, head) — reduces write scatter, improves coalescing
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int global_pos = cache_start_i + local_q_pos;
            const int block_idx = global_pos / block_size_i;
            const int block_offset = global_pos % block_size_i;
            const int physical_blk = bt_ptr[batch_idx * num_blocks_per_seq + block_idx];
            const int src_base = token_idx * src_token_stride + src_offset + head_idx * head_dim;
            const int dst_base = offset_major
                ? physical_blk * block_stride
                    + block_offset * kv_head * head_dim
                    + head_idx * head_dim
                : physical_blk * block_stride
                    + head_idx * head_size_stride
                    + block_offset * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;
                    const int src_off = src_base + dim_off;
                    const int dst_off = dst_base + dim_off;

                    bf16x8_t src8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + src_off);
                    const float* __restrict__ scale = scale_ptr == nullptr
                        ? nullptr
                        : scale_ptr + head_idx * head_dim + dim_off;

                    VecT out;
                    #pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        out.d[j] = Converter::convert(src8.d[j], scale == nullptr ? nullptr : scale + j);
                    }
                    if constexpr (StreamingWrite) {
                        streaming_store(reinterpret_cast<VecT*>(cache_ptr + dst_off), out);
                    } else {
                        *reinterpret_cast<VecT*>(cache_ptr + dst_off) = out;
                    }
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    const int src_off = src_base + i;
                    const int dst_off = dst_base + i;
                    const float* __restrict__ scale = scale_ptr == nullptr
                        ? nullptr
                        : scale_ptr + head_idx * head_dim + i;

                    auto conv_val = Converter::convert(qkv_ptr[src_off], scale);
                    if constexpr (StreamingWrite) {
                        streaming_store(cache_ptr + dst_off, conv_val);
                    } else {
                        cache_ptr[dst_off] = conv_val;
                    }
                }
            }
        });
    });
}

// ============================================================
// INT8 store: bf16 -> fp32 -> scale -> clamp -> round -> int8
// ============================================================

void store_kv_cache_int8_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor k_scale,      // [kv_head, head_dim] fp32
    torch::Tensor v_scale,      // [kv_head, head_dim] fp32
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    int8_t* __restrict__ k_cache_ptr = k_cache.data_ptr<int8_t>();
    int8_t* __restrict__ v_cache_ptr = v_cache.data_ptr<int8_t>();
    const float* __restrict__ k_scale_ptr = k_scale.data_ptr<float>();
    const float* __restrict__ v_scale_ptr = v_scale.data_ptr<float>();

    // P0: 1 WG per (token, head)
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int k_src_base = token_idx * src_token_stride + k_offset + head_idx * head_dim;
            const int v_src_base = token_idx * src_token_stride + v_offset + head_idx * head_dim;
            const int dst_base = batch_idx * cache_batch_stride
                               + head_idx * cache_head_stride
                               + (cache_start_i + local_q_pos) * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;

                    bf16x8_t kv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_base + dim_off);
                    bf16x8_t vv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_base + dim_off);

                    const float* __restrict__ ks = k_scale_ptr + head_idx * head_dim + dim_off;
                    const float* __restrict__ vs = v_scale_ptr + head_idx * head_dim + dim_off;

                    int8x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = bf16_to_fp32(kv8.d[j]) * ks[j];
                        kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                        kout.d[j] = static_cast<int8_t>(kf);

                        float vf = bf16_to_fp32(vv8.d[j]) * vs[j];
                        vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                        vout.d[j] = static_cast<int8_t>(vf);
                    }
                    *reinterpret_cast<int8x8_t*>(k_cache_ptr + dst_base + dim_off) = kout;
                    *reinterpret_cast<int8x8_t*>(v_cache_ptr + dst_base + dim_off) = vout;
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    const float ks = k_scale_ptr[head_idx * head_dim + i];
                    const float vs = v_scale_ptr[head_idx * head_dim + i];

                    float kf = bf16_to_fp32(qkv_ptr[k_src_base + i]) * ks;
                    kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                    k_cache_ptr[dst_base + i] = static_cast<int8_t>(kf);

                    float vf = bf16_to_fp32(qkv_ptr[v_src_base + i]) * vs;
                    vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                    v_cache_ptr[dst_base + i] = static_cast<int8_t>(vf);
                }
            }
        });
    });
}

// ============================================================
// BF16 store: bf16 -> bf16 permute+copy (vec8 + sub-group)
// ============================================================

void store_kv_cache_bf16_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] bf16
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] bf16
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    uint16_t* __restrict__ k_cache_ptr = reinterpret_cast<uint16_t*>(k_cache.data_ptr());
    uint16_t* __restrict__ v_cache_ptr = reinterpret_cast<uint16_t*>(v_cache.data_ptr());

    // P0: 1 WG per (token, head)
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int k_src_base = token_idx * src_token_stride + k_offset + head_idx * head_dim;
            const int v_src_base = token_idx * src_token_stride + v_offset + head_idx * head_dim;
            const int dst_base = batch_idx * cache_batch_stride
                               + head_idx * cache_head_stride
                               + (cache_start_i + local_q_pos) * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;

                    bf16x8_t k_data = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_base + dim_off);
                    bf16x8_t v_data = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_base + dim_off);
                    *reinterpret_cast<bf16x8_t*>(k_cache_ptr + dst_base + dim_off) = k_data;
                    *reinterpret_cast<bf16x8_t*>(v_cache_ptr + dst_base + dim_off) = v_data;
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    k_cache_ptr[dst_base + i] = qkv_ptr[k_src_base + i];
                    v_cache_ptr[dst_base + i] = qkv_ptr[v_src_base + i];
                }
            }
        });
    });
}

// ============================================================
// FP8 E4M3 store: bf16 -> fp32 -> scale -> clamp -> fp8_e4m3
// ============================================================

void store_kv_cache_fp8_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] uint8 (fp8)
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] uint8 (fp8)
    torch::Tensor k_scale,      // [kv_head, head_dim] fp32
    torch::Tensor v_scale,      // [kv_head, head_dim] fp32
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int n = kv_head * head_dim;
    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    uint8_t* __restrict__ k_cache_ptr = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
    uint8_t* __restrict__ v_cache_ptr = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
    const float* __restrict__ k_scale_ptr = k_scale.data_ptr<float>();
    const float* __restrict__ v_scale_ptr = v_scale.data_ptr<float>();

    // P0: 1 WG per (token, head)
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int k_src_base = token_idx * src_token_stride + k_offset + head_idx * head_dim;
            const int v_src_base = token_idx * src_token_stride + v_offset + head_idx * head_dim;
            const int dst_base = batch_idx * cache_batch_stride
                               + head_idx * cache_head_stride
                               + (cache_start_i + local_q_pos) * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;

                    bf16x8_t kv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_base + dim_off);
                    bf16x8_t vv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_base + dim_off);

                    const float* __restrict__ ks = k_scale_ptr + head_idx * head_dim + dim_off;
                    const float* __restrict__ vs = v_scale_ptr + head_idx * head_dim + dim_off;

                    uint8x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = bf16_to_fp32(kv8.d[j]) * ks[j];
                        kout.d[j] = fp32_to_fp8_e4m3(kf);

                        float vf = bf16_to_fp32(vv8.d[j]) * vs[j];
                        vout.d[j] = fp32_to_fp8_e4m3(vf);
                    }
                    *reinterpret_cast<uint8x8_t*>(k_cache_ptr + dst_base + dim_off) = kout;
                    *reinterpret_cast<uint8x8_t*>(v_cache_ptr + dst_base + dim_off) = vout;
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    const float ks = k_scale_ptr[head_idx * head_dim + i];
                    const float vs = v_scale_ptr[head_idx * head_dim + i];

                    float kf = bf16_to_fp32(qkv_ptr[k_src_base + i]) * ks;
                    k_cache_ptr[dst_base + i] = fp32_to_fp8_e4m3(kf);

                    float vf = bf16_to_fp32(qkv_ptr[v_src_base + i]) * vs;
                    v_cache_ptr[dst_base + i] = fp32_to_fp8_e4m3(vf);
                }
            }
        });
    });
}

// ============================================================
// PAGED INT8 store: bf16 -> scale -> int8, indirect via block_table
// ============================================================

void store_kv_cache_int8_paged_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [total_blocks, kv_head, block_size, head_dim] int8
    torch::Tensor v_cache,      // [total_blocks, kv_head, block_size, head_dim] int8
    torch::Tensor k_scale,      // [kv_head, head_dim] fp32
    torch::Tensor v_scale,      // [kv_head, head_dim] fp32
    torch::Tensor block_table,  // [batch, num_blocks_per_seq] int32
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int block_size_i = static_cast<int>(offset_major ? k_cache.size(1) : k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);
    const int num_blocks_per_seq = static_cast<int>(block_table.size(1));

    const int n = kv_head * head_dim;
    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int head_size_stride = block_size_i * head_dim;
    const int block_stride = kv_head * head_size_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    int8_t* __restrict__ k_cache_ptr = k_cache.data_ptr<int8_t>();
    int8_t* __restrict__ v_cache_ptr = v_cache.data_ptr<int8_t>();
    const float* __restrict__ k_scale_ptr = k_scale.data_ptr<float>();
    const float* __restrict__ v_scale_ptr = v_scale.data_ptr<float>();
    const int32_t* __restrict__ bt_ptr = block_table.data_ptr<int32_t>();

    // P0: 1 WG per (token, head)
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int global_pos = cache_start_i + local_q_pos;
            const int block_idx = global_pos / block_size_i;
            const int block_offset = global_pos % block_size_i;
            const int physical_blk = bt_ptr[batch_idx * num_blocks_per_seq + block_idx];

            const int k_src_base = token_idx * src_token_stride + k_offset + head_idx * head_dim;
            const int v_src_base = token_idx * src_token_stride + v_offset + head_idx * head_dim;
            const int dst_base = offset_major
                ? physical_blk * block_stride
                    + block_offset * kv_head * head_dim
                    + head_idx * head_dim
                : physical_blk * block_stride
                    + head_idx * head_size_stride
                    + block_offset * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;

                    bf16x8_t kv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_base + dim_off);
                    bf16x8_t vv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_base + dim_off);

                    const float* __restrict__ ks = k_scale_ptr + head_idx * head_dim + dim_off;
                    const float* __restrict__ vs = v_scale_ptr + head_idx * head_dim + dim_off;

                    int8x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = bf16_to_fp32(kv8.d[j]) * ks[j];
                        kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                        kout.d[j] = static_cast<int8_t>(kf);

                        float vf = bf16_to_fp32(vv8.d[j]) * vs[j];
                        vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                        vout.d[j] = static_cast<int8_t>(vf);
                    }
                    *reinterpret_cast<int8x8_t*>(k_cache_ptr + dst_base + dim_off) = kout;
                    *reinterpret_cast<int8x8_t*>(v_cache_ptr + dst_base + dim_off) = vout;
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    const float ks = k_scale_ptr[head_idx * head_dim + i];
                    const float vs = v_scale_ptr[head_idx * head_dim + i];

                    float kf = bf16_to_fp32(qkv_ptr[k_src_base + i]) * ks;
                    kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                    k_cache_ptr[dst_base + i] = static_cast<int8_t>(kf);

                    float vf = bf16_to_fp32(qkv_ptr[v_src_base + i]) * vs;
                    vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                    v_cache_ptr[dst_base + i] = static_cast<int8_t>(vf);
                }
            }
        });
    });
}

// ============================================================
// PAGED BF16 store: bf16 -> bf16 permute+copy, indirect via block_table
// ============================================================

void store_kv_cache_bf16_paged_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [total_blocks, kv_head, block_size, head_dim] bf16
    torch::Tensor v_cache,      // [total_blocks, kv_head, block_size, head_dim] bf16
    torch::Tensor block_table,  // [batch, num_blocks_per_seq] int32
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int block_size_i = static_cast<int>(offset_major ? k_cache.size(1) : k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);
    const int num_blocks_per_seq = static_cast<int>(block_table.size(1));

    const int n = kv_head * head_dim;
    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int head_size_stride = block_size_i * head_dim;
    const int block_stride = kv_head * head_size_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    uint16_t* __restrict__ k_cache_ptr = reinterpret_cast<uint16_t*>(k_cache.data_ptr());
    uint16_t* __restrict__ v_cache_ptr = reinterpret_cast<uint16_t*>(v_cache.data_ptr());
    const int32_t* __restrict__ bt_ptr = block_table.data_ptr<int32_t>();

    // P0: 1 WG per (token, head)
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int global_pos = cache_start_i + local_q_pos;
            const int block_idx = global_pos / block_size_i;
            const int block_offset = global_pos % block_size_i;
            const int physical_blk = bt_ptr[batch_idx * num_blocks_per_seq + block_idx];

            const int k_src_base = token_idx * src_token_stride + k_offset + head_idx * head_dim;
            const int v_src_base = token_idx * src_token_stride + v_offset + head_idx * head_dim;
            const int dst_base = offset_major
                ? physical_blk * block_stride
                    + block_offset * kv_head * head_dim
                    + head_idx * head_dim
                : physical_blk * block_stride
                    + head_idx * head_size_stride
                    + block_offset * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;

                    bf16x8_t k_data = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_base + dim_off);
                    bf16x8_t v_data = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_base + dim_off);
                    *reinterpret_cast<bf16x8_t*>(k_cache_ptr + dst_base + dim_off) = k_data;
                    *reinterpret_cast<bf16x8_t*>(v_cache_ptr + dst_base + dim_off) = v_data;
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    k_cache_ptr[dst_base + i] = qkv_ptr[k_src_base + i];
                    v_cache_ptr[dst_base + i] = qkv_ptr[v_src_base + i];
                }
            }
        });
    });
}

// ============================================================
// PAGED FP8 E4M3 store: bf16 -> scale -> fp8, indirect via block_table
// ============================================================

void store_kv_cache_fp8_paged_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [total_blocks, kv_head, block_size, head_dim] uint8 (fp8)
    torch::Tensor v_cache,      // [total_blocks, kv_head, block_size, head_dim] uint8 (fp8)
    torch::Tensor k_scale,      // [kv_head, head_dim] fp32
    torch::Tensor v_scale,      // [kv_head, head_dim] fp32
    torch::Tensor block_table,  // [batch, num_blocks_per_seq] int32
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int block_size_i = static_cast<int>(offset_major ? k_cache.size(1) : k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);
    const int num_blocks_per_seq = static_cast<int>(block_table.size(1));

    const int n = kv_head * head_dim;
    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int head_size_stride = block_size_i * head_dim;
    const int block_stride = kv_head * head_size_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    uint8_t* __restrict__ k_cache_ptr = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
    uint8_t* __restrict__ v_cache_ptr = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
    const float* __restrict__ k_scale_ptr = k_scale.data_ptr<float>();
    const float* __restrict__ v_scale_ptr = v_scale.data_ptr<float>();
    const int32_t* __restrict__ bt_ptr = block_table.data_ptr<int32_t>();

    // P0: 1 WG per (token, head)
    const int num_groups = num_tokens * kv_head;
    const int wg_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_id = item.get_group(0);
            const int token_idx = group_id / kv_head;
            const int head_idx = group_id % kv_head;
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int global_pos = cache_start_i + local_q_pos;
            const int block_idx = global_pos / block_size_i;
            const int block_offset = global_pos % block_size_i;
            const int physical_blk = bt_ptr[batch_idx * num_blocks_per_seq + block_idx];

            const int k_src_base = token_idx * src_token_stride + k_offset + head_idx * head_dim;
            const int v_src_base = token_idx * src_token_stride + v_offset + head_idx * head_dim;
            const int dst_base = offset_major
                ? physical_blk * block_stride
                    + block_offset * kv_head * head_dim
                    + head_idx * head_dim
                : physical_blk * block_stride
                    + head_idx * head_size_stride
                    + block_offset * head_dim;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = head_dim / VEC;
                for (int vi = lid; vi < total_vec; vi += wg_size) {
                    const int dim_off = vi * VEC;

                    bf16x8_t kv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_base + dim_off);
                    bf16x8_t vv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_base + dim_off);

                    const float* __restrict__ ks = k_scale_ptr + head_idx * head_dim + dim_off;
                    const float* __restrict__ vs = v_scale_ptr + head_idx * head_dim + dim_off;

                    uint8x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = bf16_to_fp32(kv8.d[j]) * ks[j];
                        kout.d[j] = fp32_to_fp8_e4m3(kf);

                        float vf = bf16_to_fp32(vv8.d[j]) * vs[j];
                        vout.d[j] = fp32_to_fp8_e4m3(vf);
                    }
                    *reinterpret_cast<uint8x8_t*>(k_cache_ptr + dst_base + dim_off) = kout;
                    *reinterpret_cast<uint8x8_t*>(v_cache_ptr + dst_base + dim_off) = vout;
                }
            } else {
                for (int i = lid; i < head_dim; i += wg_size) {
                    const float ks = k_scale_ptr[head_idx * head_dim + i];
                    const float vs = v_scale_ptr[head_idx * head_dim + i];

                    float kf = bf16_to_fp32(qkv_ptr[k_src_base + i]) * ks;
                    k_cache_ptr[dst_base + i] = fp32_to_fp8_e4m3(kf);

                    float vf = bf16_to_fp32(qkv_ptr[v_src_base + i]) * vs;
                    v_cache_ptr[dst_base + i] = fp32_to_fp8_e4m3(vf);
                }
            }
        });
    });
}

void store_kv_cache_int8_single_sycl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    torch::Tensor scale,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    store_kv_cache_single_linear_impl<int8_t, int8x8_t, Int8Converter, true>(
        packed_qkv, cache, scale.data_ptr<float>(), src_head_start, kv_head_num,
        batch_size, q_len, cache_start);
}

void store_kv_cache_bf16_single_sycl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    store_kv_cache_single_linear_impl<uint16_t, bf16x8_t, Bf16CopyConverter>(
        packed_qkv, cache, nullptr, src_head_start, kv_head_num,
        batch_size, q_len, cache_start);
}

void store_kv_cache_fp8_single_sycl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    torch::Tensor scale,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    store_kv_cache_single_linear_impl<uint8_t, uint8x8_t, Fp8Converter>(
        packed_qkv, cache, scale.data_ptr<float>(), src_head_start, kv_head_num,
        batch_size, q_len, cache_start);
}

void store_kv_cache_int8_single_paged_sycl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    torch::Tensor scale,
    torch::Tensor block_table,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    store_kv_cache_single_paged_impl<int8_t, int8x8_t, Int8Converter, true>(
        packed_qkv, cache, scale.data_ptr<float>(), block_table, src_head_start,
        kv_head_num, batch_size, q_len, cache_start, offset_major);
}

void store_kv_cache_bf16_single_paged_sycl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    torch::Tensor block_table,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    store_kv_cache_single_paged_impl<uint16_t, bf16x8_t, Bf16CopyConverter>(
        packed_qkv, cache, nullptr, block_table, src_head_start, kv_head_num,
        batch_size, q_len, cache_start, offset_major);
}

void store_kv_cache_fp8_single_paged_sycl(
    torch::Tensor packed_qkv,
    torch::Tensor cache,
    torch::Tensor scale,
    torch::Tensor block_table,
    int64_t src_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start,
    bool offset_major
) {
    store_kv_cache_single_paged_impl<uint8_t, uint8x8_t, Fp8Converter>(
        packed_qkv, cache, scale.data_ptr<float>(), block_table, src_head_start,
        kv_head_num, batch_size, q_len, cache_start, offset_major);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("store_kv_cache_int8", &store_kv_cache_int8_sycl,
          "Fused bf16->int8 quantized store to linear KV cache (SYCL v5)");
    m.def("store_kv_cache_bf16", &store_kv_cache_bf16_sycl,
          "Fused bf16->bf16 store to linear KV cache (SYCL v5)");
    m.def("store_kv_cache_fp8", &store_kv_cache_fp8_sycl,
          "Fused bf16->fp8_e4m3 quantized store to linear KV cache (SYCL)");
        m.def("store_kv_cache_int8_single", &store_kv_cache_int8_single_sycl,
            "Fused bf16->int8 quantized store to one linear KV cache stream (SYCL)");
        m.def("store_kv_cache_bf16_single", &store_kv_cache_bf16_single_sycl,
            "Fused bf16->bf16 store to one linear KV cache stream (SYCL)");
        m.def("store_kv_cache_fp8_single", &store_kv_cache_fp8_single_sycl,
            "Fused bf16->fp8_e4m3 quantized store to one linear KV cache stream (SYCL)");
    m.def("store_kv_cache_int8_paged", &store_kv_cache_int8_paged_sycl,
          "Fused bf16->int8 quantized store to paged KV cache (SYCL)");
    m.def("store_kv_cache_bf16_paged", &store_kv_cache_bf16_paged_sycl,
          "Fused bf16->bf16 store to paged KV cache (SYCL)");
    m.def("store_kv_cache_fp8_paged", &store_kv_cache_fp8_paged_sycl,
          "Fused bf16->fp8_e4m3 quantized store to paged KV cache (SYCL)");
        m.def("store_kv_cache_int8_single_paged", &store_kv_cache_int8_single_paged_sycl,
            "Fused bf16->int8 quantized store to one paged KV cache stream (SYCL)");
        m.def("store_kv_cache_bf16_single_paged", &store_kv_cache_bf16_single_paged_sycl,
            "Fused bf16->bf16 store to one paged KV cache stream (SYCL)");
        m.def("store_kv_cache_fp8_single_paged", &store_kv_cache_fp8_single_paged_sycl,
            "Fused bf16->fp8_e4m3 quantized store to one paged KV cache stream (SYCL)");
}
