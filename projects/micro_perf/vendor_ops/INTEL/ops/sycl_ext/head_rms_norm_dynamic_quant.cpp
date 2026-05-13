// Head RMS Norm + Smooth Scale + Per-Token Dynamic Quant (int8) — SYCL Extension.
//
// Algorithm (per token t):
//   1. For each head h: var_h = mean(X[t,h,:]^2); denom_h = rsqrt(var_h + eps)
//   2. normed[t,h,d] = X[t,h,d] * denom_h * norm_weight[d]
//   3. smoothed[t,i]  = normed[t,i]  * smooth_scale[i]      (i = h*head_dim+d)
//   4. abs_max_t       = max_i |smoothed[t,i]|
//   5. quant[t,i]      = round(clamp(smoothed[t,i] * 127 / abs_max_t, -127, 127))
//   6. per_token_scale[t] = abs_max_t / 127  (dequant factor)

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>
#include <tuple>

namespace {

namespace py = pybind11;

constexpr int SIMD = 16;

struct SyclKerConfigBase {};
#define __SYCL_KER_CONFIG_CONVENTION__ SyclKerConfigBase
#define SYCL_REQD_SUB_GROUP_SIZE(SIZE) [[sycl::reqd_sub_group_size(SIZE)]]

template <typename T>
using sycl_local_acc_t = sycl::local_accessor<T, 1>;

template <typename out_t>
inline out_t quant_convert(float q_f) {
  if constexpr (std::is_same_v<out_t, int8_t>) {
    q_f = sycl::round(q_f);
    return static_cast<int8_t>(static_cast<int>(q_f));
  } else {
    return out_t(q_f);
  }
}
// ---------------------------------------------------------------------------
// Optimized fused kernel for head_dim >= 32 (Medium path).
//
// Strategy:
//   - TPB=256 threads per workgroup, SIMD=16 → NWARPS=16 subgroups.
//   - Each subgroup processes heads in a strided loop: subgroup `wid` owns
//     heads wid, wid+NWARPS, wid+2*NWARPS, ...
//   - For each owned head, each lane loads ELEMS_PER_LOAD=4/sizeof(T) elements,
//     repeated UNROLL times to cover head_dim. Data is cached in private
//     registers `T regs[UNROLL * ELEMS_PER_LOAD]`.
//   - Phase 1: subgroup-reduce var_sum → per-head denom stored in denom_buf_.
//   - Phase 2: reuse regs (no re-read), compute per-head abs_max, reduce
//     across all heads via warp_buf_.
//   - Phase 3: reuse regs again to emit int8 output.
//
// Register usage per thread (example): UNROLL=3, ELEMS_PER_LOAD=2 → 6 regs.
// For large head_num the subgroup loops over multiple heads, reusing registers.
//
// Constraint: UNROLL = ceil(head_dim / (SIMD * ELEMS_PER_LOAD))
//   → head_dim=96, bf16: UNROLL=ceil(96/32)=3.
// ---------------------------------------------------------------------------
template <typename T, typename out_t, int TPB, int UNROLL, int BPL = 4>
struct HeadRMSNormDynamicQuantMediumFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static_assert(BPL == 4 || BPL == 8 || BPL == 16, "BPL must be 4, 8, or 16");
  // Number of T elements loaded per lane per step.
  static constexpr int ELEMS_PER_LOAD = BPL / static_cast<int>(sizeof(T));
  static constexpr int NWARPS = TPB / SIMD;
  using payload_t = typename std::conditional<BPL == 4, uint32_t,
                      typename std::conditional<BPL == 8, sycl::vec<uint32_t, 2>,
                                                          sycl::vec<uint32_t, 4>
                      >::type>::type;
  using pack_t = typename std::conditional<ELEMS_PER_LOAD == 8, uint64_t,
                   typename std::conditional<ELEMS_PER_LOAD == 4, uint32_t,
                     typename std::conditional<ELEMS_PER_LOAD == 2, uint16_t,
                                                                    uint8_t
                     >::type>::type>::type;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int token_idx = static_cast<int>(item_id.get_group(0));
    const int tid       = static_cast<int>(item_id.get_local_id(0));

    if (token_idx >= num_tokens_) return;

    auto sg           = item_id.get_sub_group();
    const int sg_lane = static_cast<int>(sg.get_local_linear_id());
    const int wid     = tid / SIMD;   // subgroup index within workgroup

    const int N_total  = head_num_ * head_dim_;
    const T*    row    = X_        + static_cast<uint64_t>(token_idx) * N_total;
    out_t* out_row     = quant_out_ + static_cast<uint64_t>(token_idx) * N_total;

    // ── Phase 1: Per-head variance → store denom in denom_buf_ ───────────
    // Each subgroup processes heads: wid, wid+NWARPS, wid+2*NWARPS, ...
    for (int h = wid; h < head_num_; h += NWARPS) {
      const payload_t* pptr = reinterpret_cast<const payload_t*>(
          row + static_cast<uint64_t>(h) * head_dim_);

      // Private register cache for this head's elements owned by this lane.
      alignas(BPL) T regs[UNROLL * ELEMS_PER_LOAD];
      payload_t* regs_p = reinterpret_cast<payload_t*>(regs);
      float var_sum = 0.f;

#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        sycl::ext::oneapi::experimental::group_load(
        sg, pptr + u * SIMD, regs_p[u]);

        const int k0 = (u * SIMD + sg_lane) * ELEMS_PER_LOAD;
#pragma unroll
        for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
          const int k = k0 + e;
          const T val = regs[u * ELEMS_PER_LOAD + e];
          var_sum += static_cast<float>(val) * static_cast<float>(val);
        }
      }

      // Subgroup reduce var_sum.
      var_sum = sycl::reduce_over_group(sg, var_sum, sycl::plus<float>());
      const float denom = sycl::rsqrt(
          var_sum / static_cast<float>(head_dim_) + epsilon_);

      // Reuse already-loaded regs to compute per-head abs_max (no global reload).
      float abs_max = 0.f;
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int base = (u * SIMD + sg_lane) * ELEMS_PER_LOAD;
#pragma unroll
        for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
          const int k = base + e;
          const float val     = static_cast<float>(regs[u * ELEMS_PER_LOAD + e]);
          const int   gk      = h * head_dim_ + k;
          const float normed  = val * denom;
          const float smoothed= normed * norm_weight_[k] * smooth_scale_[gk];
          abs_max = sycl::fmax(abs_max, sycl::fabs(smoothed));
        }
      }

      // Subgroup reduce abs_max for this head.
      abs_max = sycl::reduce_over_group(sg, abs_max, sycl::maximum<float>());

      // Store denom for later phases (one slot per head)
      if (sg_lane == 0) denom_buf_[h] = denom;
      if (sg_lane == 0) warp_buf_[h] = abs_max;
    }
    sycl::group_barrier(item_id.get_group());

    // ── Phase 2: per-token abs_max reduction over heads ───────────────────

    // Cross-head reduce abs_max (all threads participate, strided)
    float token_abs_max = 0.f;
    for (int h = tid; h < head_num_; h += TPB)
      token_abs_max = sycl::fmax(token_abs_max, warp_buf_[h]);

    // Workgroup-level reduce
    warp_buf_[tid] = token_abs_max;
    sycl::group_barrier(item_id.get_group());
    // Binary tree reduce within warp_buf_[0..TPB-1]
    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        warp_buf_[tid] = sycl::fmax(warp_buf_[tid], warp_buf_[tid + stride]);
      sycl::group_barrier(item_id.get_group());
    }
    const float abs_max = warp_buf_[0];

    if (tid == 0)
      per_token_scale_[token_idx] = abs_max / max_dtype_val_;

    // ── Phase 3: Quantize using reloaded registers ─────────────────────────
    const float inv_scale = (abs_max > 0.f) ? (max_dtype_val_ / abs_max) : 0.f;
    for (int h = wid; h < head_num_; h += NWARPS) {
      const payload_t* pptr = reinterpret_cast<const payload_t*>(
          row + static_cast<uint64_t>(h) * head_dim_);

      // Only one ELEMS_PER_LOAD slot needed: loaded and consumed within the
      // same 'u' iteration, so no need to hold all UNROLL slices at once.
      alignas(BPL) T regs[ELEMS_PER_LOAD];
      payload_t* regs_p = reinterpret_cast<payload_t*>(regs);

      const float denom = denom_buf_[h];
      out_t* head_out   = out_row + static_cast<uint64_t>(h) * head_dim_;

#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        sycl::ext::oneapi::experimental::group_load(
            sg, pptr + u * SIMD, regs_p[0]);

        const int base = (u * SIMD + sg_lane) * ELEMS_PER_LOAD;
        if constexpr (std::is_same_v<out_t, int8_t>) {
          pack_t out_pack = 0;
#pragma unroll
          for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
            const int k = base + e;
            const float val     = static_cast<float>(regs[e]);
            const int   gk      = h * head_dim_ + k;
            const float normed  = val * denom;
            const float smoothed= normed * norm_weight_[k] * smooth_scale_[gk];
            float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
            const int8_t qv = quant_convert<int8_t>(q_f);
            out_pack |= static_cast<pack_t>(static_cast<uint8_t>(qv)) << (e * 8);
          }
          pack_t* out_ptr = reinterpret_cast<pack_t*>(head_out);
          sycl::ext::oneapi::experimental::group_store(sg, out_pack, out_ptr + u * SIMD);
        } else {
#pragma unroll
          for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
            const int k = base + e;
            const float val     = static_cast<float>(regs[e]);
            const int   gk      = h * head_dim_ + k;
            const float normed  = val * denom;
            const float smoothed= normed * norm_weight_[k] * smooth_scale_[gk];
            float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
            head_out[k] = quant_convert<out_t>(q_f);
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    // denom_buf_: one float per head for inter-phase communication
    denom_buf_ = sycl_local_acc_t<float>(static_cast<size_t>(head_num_), cgh);
    // warp_buf_: max(head_num, TPB) slots; first used per-head, then workgroup reduce
    const size_t wb_size = static_cast<size_t>(head_num_ > TPB ? head_num_ : TPB);
    warp_buf_ = sycl_local_acc_t<float>(wb_size, cgh);
  }

  HeadRMSNormDynamicQuantMediumFunctor(
      int num_tokens, int head_num, int head_dim,
      float epsilon, float max_dtype_val,
      const T* X, const float* norm_weight, const float* smooth_scale,
      out_t* quant_out, float* per_token_scale)
      : num_tokens_(num_tokens), head_num_(head_num), head_dim_(head_dim),
        epsilon_(epsilon), max_dtype_val_(max_dtype_val),
        X_(X), norm_weight_(norm_weight), smooth_scale_(smooth_scale),
        quant_out_(quant_out), per_token_scale_(per_token_scale) {}

 private:
  int   num_tokens_, head_num_, head_dim_;
  float epsilon_, max_dtype_val_;
  const T*     X_;
  const float* norm_weight_;
  const float* smooth_scale_;
  out_t*       quant_out_;
  float*       per_token_scale_;
  sycl_local_acc_t<float> denom_buf_;
  sycl_local_acc_t<float> warp_buf_;
};

// ---------------------------------------------------------------------------
// Big fused kernel for very large head_dim or non-divisible tail cases.
//
// Compared with Medium path:
//   - Keeps the same BPL (4/8/16) vectorized subgroup load/store strategy.
//   - Uses runtime loops instead of compile-time UNROLL specialization.
//   - Handles tail elements safely when head_dim is not divisible by STEP.
// ---------------------------------------------------------------------------
template <typename T, typename out_t, int TPB, int BPL = 4>
struct HeadRMSNormDynamicQuantBigFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static_assert(BPL == 4 || BPL == 8 || BPL == 16, "BPL must be 4, 8, or 16");
  static constexpr int ELEMS_PER_LOAD = BPL / static_cast<int>(sizeof(T));
  static constexpr int NWARPS = TPB / SIMD;
  using payload_t = typename std::conditional<BPL == 4, uint32_t,
                      typename std::conditional<BPL == 8, sycl::vec<uint32_t, 2>,
                                                          sycl::vec<uint32_t, 4>
                      >::type>::type;
  using pack_t = typename std::conditional<ELEMS_PER_LOAD == 8, uint64_t,
                   typename std::conditional<ELEMS_PER_LOAD == 4, uint32_t,
                     typename std::conditional<ELEMS_PER_LOAD == 2, uint16_t,
                                                                    uint8_t
                     >::type>::type>::type;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int token_idx = static_cast<int>(item_id.get_group(0));
    const int tid       = static_cast<int>(item_id.get_local_id(0));

    if (token_idx >= num_tokens_) return;

    auto sg           = item_id.get_sub_group();
    const int sg_lane = static_cast<int>(sg.get_local_linear_id());
    const int wid     = tid / SIMD;

    const int N_total = head_num_ * head_dim_;
    const T* row      = X_ + static_cast<uint64_t>(token_idx) * N_total;
    out_t* out_row    = quant_out_ + static_cast<uint64_t>(token_idx) * N_total;

    constexpr int STEP = SIMD * ELEMS_PER_LOAD;
    const int full_blocks = head_dim_ / STEP;
    const int tail_base = full_blocks * STEP;

    // ── Phase 1: per-head variance + per-head abs_max ───────────────────
    for (int h = wid; h < head_num_; h += NWARPS) {
      const T* head_row = row + static_cast<uint64_t>(h) * head_dim_;
      const payload_t* pptr = reinterpret_cast<const payload_t*>(head_row);

      alignas(BPL) T regs[ELEMS_PER_LOAD];
      payload_t* regs_p = reinterpret_cast<payload_t*>(regs);
      float var_sum = 0.f;

      for (int u = 0; u < full_blocks; ++u) {
        sycl::ext::oneapi::experimental::group_load(
            sg, pptr + u * SIMD, regs_p[0]);

        const int k0 = (u * SIMD + sg_lane) * ELEMS_PER_LOAD;
#pragma unroll
        for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
          const T val = regs[e];
          var_sum += static_cast<float>(val) * static_cast<float>(val);
        }
      }

#pragma unroll
      for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
        const int k = tail_base + sg_lane * ELEMS_PER_LOAD + e;
        if (k < head_dim_) {
          const T val = head_row[k];
          var_sum += static_cast<float>(val) * static_cast<float>(val);
        }
      }

      var_sum = sycl::reduce_over_group(sg, var_sum, sycl::plus<float>());
      const float denom = sycl::rsqrt(
          var_sum / static_cast<float>(head_dim_) + epsilon_);

      float abs_max = 0.f;
      for (int u = 0; u < full_blocks; ++u) {
        sycl::ext::oneapi::experimental::group_load(
            sg, pptr + u * SIMD, regs_p[0]);

        const int base = (u * SIMD + sg_lane) * ELEMS_PER_LOAD;
#pragma unroll
        for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
          const int k = base + e;
          const int gk = h * head_dim_ + k;
          const float normed = static_cast<float>(regs[e]) * denom;
          const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
          abs_max = sycl::fmax(abs_max, sycl::fabs(smoothed));
        }
      }

#pragma unroll
      for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
        const int k = tail_base + sg_lane * ELEMS_PER_LOAD + e;
        if (k < head_dim_) {
          const int gk = h * head_dim_ + k;
          const float normed = static_cast<float>(head_row[k]) * denom;
          const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
          abs_max = sycl::fmax(abs_max, sycl::fabs(smoothed));
        }
      }

      abs_max = sycl::reduce_over_group(sg, abs_max, sycl::maximum<float>());

      if (sg_lane == 0) denom_buf_[h] = denom;
      if (sg_lane == 0) warp_buf_[h] = abs_max;
    }
    sycl::group_barrier(item_id.get_group());

    // ── Phase 2: per-token abs_max reduction over heads ─────────────────
    float token_abs_max = 0.f;
    for (int h = tid; h < head_num_; h += TPB)
      token_abs_max = sycl::fmax(token_abs_max, warp_buf_[h]);

    warp_buf_[tid] = token_abs_max;
    sycl::group_barrier(item_id.get_group());
    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        warp_buf_[tid] = sycl::fmax(warp_buf_[tid], warp_buf_[tid + stride]);
      sycl::group_barrier(item_id.get_group());
    }
    const float abs_max = warp_buf_[0];

    if (tid == 0)
      per_token_scale_[token_idx] = abs_max / max_dtype_val_;

    // ── Phase 3: quantize ────────────────────────────────────────────────
    const float inv_scale = (abs_max > 0.f) ? (max_dtype_val_ / abs_max) : 0.f;
    for (int h = wid; h < head_num_; h += NWARPS) {
      const T* head_row = row + static_cast<uint64_t>(h) * head_dim_;
      const payload_t* pptr = reinterpret_cast<const payload_t*>(head_row);

      alignas(BPL) T regs[ELEMS_PER_LOAD];
      payload_t* regs_p = reinterpret_cast<payload_t*>(regs);

      const float denom = denom_buf_[h];
      out_t* head_out = out_row + static_cast<uint64_t>(h) * head_dim_;

      for (int u = 0; u < full_blocks; ++u) {
        sycl::ext::oneapi::experimental::group_load(
            sg, pptr + u * SIMD, regs_p[0]);

        const int base = (u * SIMD + sg_lane) * ELEMS_PER_LOAD;
        if constexpr (std::is_same_v<out_t, int8_t>) {
          pack_t out_pack = 0;
#pragma unroll
          for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
            const int k = base + e;
            const int gk = h * head_dim_ + k;
            const float normed = static_cast<float>(regs[e]) * denom;
            const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
            float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
            const int8_t qv = quant_convert<int8_t>(q_f);
            out_pack |= static_cast<pack_t>(static_cast<uint8_t>(qv)) << (e * 8);
          }
          pack_t* out_ptr = reinterpret_cast<pack_t*>(head_out);
          sycl::ext::oneapi::experimental::group_store(sg, out_pack, out_ptr + u * SIMD);
        } else {
#pragma unroll
          for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
            const int k = base + e;
            const int gk = h * head_dim_ + k;
            const float normed = static_cast<float>(regs[e]) * denom;
            const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
            float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
            head_out[k] = quant_convert<out_t>(q_f);
          }
        }
      }

#pragma unroll
      for (int e = 0; e < ELEMS_PER_LOAD; ++e) {
        const int k = tail_base + sg_lane * ELEMS_PER_LOAD + e;
        if (k < head_dim_) {
          const int gk = h * head_dim_ + k;
          const float normed = static_cast<float>(head_row[k]) * denom;
          const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
          float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
          head_out[k] = quant_convert<out_t>(q_f);
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    denom_buf_ = sycl_local_acc_t<float>(static_cast<size_t>(head_num_), cgh);
    const size_t wb_size = static_cast<size_t>(head_num_ > TPB ? head_num_ : TPB);
    warp_buf_ = sycl_local_acc_t<float>(wb_size, cgh);
  }

  HeadRMSNormDynamicQuantBigFunctor(
      int num_tokens, int head_num, int head_dim,
      float epsilon, float max_dtype_val,
      const T* X, const float* norm_weight, const float* smooth_scale,
      out_t* quant_out, float* per_token_scale)
      : num_tokens_(num_tokens), head_num_(head_num), head_dim_(head_dim),
        epsilon_(epsilon), max_dtype_val_(max_dtype_val),
        X_(X), norm_weight_(norm_weight), smooth_scale_(smooth_scale),
        quant_out_(quant_out), per_token_scale_(per_token_scale) {}

 private:
  int   num_tokens_, head_num_, head_dim_;
  float epsilon_, max_dtype_val_;
  const T*     X_;
  const float* norm_weight_;
  const float* smooth_scale_;
  out_t*       quant_out_;
  float*       per_token_scale_;
  sycl_local_acc_t<float> denom_buf_;
  sycl_local_acc_t<float> warp_buf_;
};

// ---------------------------------------------------------------------------
// Specialized small kernel for HD=8/16.
// - Caches full token row in SLM via subgroup vectorized loads.
// - Uses compile-time HD specialization and packed subgroup stores.
// ---------------------------------------------------------------------------
template <typename T, typename out_t, int TPB, int HD>
struct HeadRMSNormDynamicQuantSmallSpecializedFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int NWARPS = TPB / SIMD;
  using vec4u32 = sycl::vec<uint32_t, 4>;
  using out8_t = sycl::vec<uint32_t, 2>;
  using out16_t = sycl::vec<uint16_t, 8>;
  using simd8_t = sycl::vec<int8_t, 8>;
  using simd16_t = sycl::vec<int8_t, 16>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int token_idx = static_cast<int>(item_id.get_group(0));
    const int tid       = static_cast<int>(item_id.get_local_id(0));
    if (token_idx >= num_tokens_) return;

    const int N_total  = head_num_ * HD;
    // 强制对齐输入输出指针，避免misaligned penalty
    const T* __restrict__ row       = reinterpret_cast<const T*>(__builtin_assume_aligned(X_ + static_cast<int64_t>(token_idx) * N_total, 16));
    out_t* __restrict__ out_row     = reinterpret_cast<out_t*>(__builtin_assume_aligned(quant_out_ + static_cast<int64_t>(token_idx) * N_total, 16));

    auto sg            = item_id.get_sub_group();
    const int sg_lane  = static_cast<int>(sg.get_local_linear_id());
    const int wid      = tid / SIMD;

    T* raw_row = raw_buf_.template get_multi_ptr<sycl::access::decorated::no>().get();

    // SLM加载向量化优化，block copy
    const int total_bytes = N_total * static_cast<int>(sizeof(T));
    const int vec_chunks = total_bytes / (SIMD * 16);
    const int vec_tail = total_bytes % (SIMD * 16);

    const vec4u32* src_v = reinterpret_cast<const vec4u32*>(row);
    vec4u32* dst_v = reinterpret_cast<vec4u32*>(raw_row);
    // SLM加载：先处理完整chunk，再处理tail，避免未初始化
    for (int c = wid; c < vec_chunks; c += NWARPS) {
      vec4u32 v;
      sycl::ext::oneapi::experimental::group_load(sg, src_v + c * SIMD, v);
      sycl::ext::oneapi::experimental::group_store(sg, v, dst_v + c * SIMD);
    }
    // tail处理：所有线程都参与，避免遗漏
    if (vec_tail > 0) {
      const uint8_t* src_b = reinterpret_cast<const uint8_t*>(row);
      uint8_t* dst_b = reinterpret_cast<uint8_t*>(raw_row);
      const int tail_base = vec_chunks * SIMD * 16;
      for (int b = tid; b < vec_tail; b += TPB)
        dst_b[tail_base + b] = src_b[tail_base + b];
    }
    sycl::group_barrier(item_id.get_group());

    for (int h_base = wid * SIMD; h_base < head_num_; h_base += NWARPS * SIMD) {
      const int h = h_base + sg_lane;
      // 分支消除：用掩码
      if (h < head_num_) {
        const T* head_row = raw_row + static_cast<int64_t>(h) * HD;
        float var_sum = 0.f;
#pragma unroll
        for (int k = 0; k < HD; ++k) {
          const float v = static_cast<float>(head_row[k]);
          var_sum += v * v;
        }
        const float denom = sycl::rsqrt(var_sum / static_cast<float>(HD) + epsilon_);
        float head_abs_max = 0.f;
#pragma unroll
        for (int k = 0; k < HD; ++k) {
          const int gk = h * HD + k;
          const float normed = static_cast<float>(head_row[k]) * denom;
          const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
          head_abs_max = sycl::fmax(head_abs_max, sycl::fabs(smoothed));
        }
        denom_buf_[h] = denom;
        warp_buf_[h] = head_abs_max;
      }
    }
    sycl::group_barrier(item_id.get_group());

    // 分支消除+SIMD规约
    float token_abs_max = 0.f;
    for (int h = tid; h < head_num_; h += TPB)
      token_abs_max = sycl::fmax(token_abs_max, warp_buf_[h]);
    warp_buf_[tid] = token_abs_max;
    sycl::group_barrier(item_id.get_group());
    // 规约时只用tid==0线程写per_token_scale，避免未初始化
    float abs_max = warp_buf_[0];
    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        warp_buf_[tid] = sycl::fmax(warp_buf_[tid], warp_buf_[tid + stride]);
      sycl::group_barrier(item_id.get_group());
    }
    abs_max = warp_buf_[0];
    if (tid == 0)
      per_token_scale_[token_idx] = abs_max / max_dtype_val_;

    const float inv_scale = (abs_max > 0.f) ? (max_dtype_val_ / abs_max) : 0.f;
    for (int h_base = wid * SIMD; h_base < head_num_; h_base += NWARPS * SIMD) {
      const int h = h_base + sg_lane;
      const bool full_chunk = (h_base + SIMD <= head_num_);
      const bool valid_head = (h < head_num_);
      if constexpr (HD == 8) {
        if constexpr (std::is_same_v<out_t, int8_t>) {
        out8_t out_pack;
        simd8_t qv(0);
        if (valid_head) {
          const T* head_row = raw_row + static_cast<int64_t>(h) * 8;
          const float denom = denom_buf_[h];
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int gk = h * 8 + k;
            const float normed = static_cast<float>(head_row[k]) * denom;
            const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
            float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
            qv[k] = quant_convert<int8_t>(q_f);
          }
        }
        // SIMD打包
        out_pack[0] =
            static_cast<uint32_t>(static_cast<uint8_t>(qv[0])) |
            (static_cast<uint32_t>(static_cast<uint8_t>(qv[1])) << 8) |
            (static_cast<uint32_t>(static_cast<uint8_t>(qv[2])) << 16) |
            (static_cast<uint32_t>(static_cast<uint8_t>(qv[3])) << 24);
        out_pack[1] =
            static_cast<uint32_t>(static_cast<uint8_t>(qv[4])) |
            (static_cast<uint32_t>(static_cast<uint8_t>(qv[5])) << 8) |
            (static_cast<uint32_t>(static_cast<uint8_t>(qv[6])) << 16) |
            (static_cast<uint32_t>(static_cast<uint8_t>(qv[7])) << 24);
        out8_t* out_ptr = reinterpret_cast<out8_t*>(out_row + static_cast<int64_t>(h_base) * 8);
        if (full_chunk) {
          sycl::ext::oneapi::experimental::group_store(sg, out_pack, out_ptr);
        } else if (valid_head) {
          out_ptr[sg_lane] = out_pack;
        }
        } else {
          if (valid_head) {
            const T* head_row = raw_row + static_cast<int64_t>(h) * 8;
            const float denom = denom_buf_[h];
            out_t* out_ptr = out_row + static_cast<int64_t>(h) * 8;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
              const int gk = h * 8 + k;
              const float normed = static_cast<float>(head_row[k]) * denom;
              const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
              float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
              out_ptr[k] = quant_convert<out_t>(q_f);
            }
          }
        }
      } else {
        if constexpr (std::is_same_v<out_t, int8_t>) {
          out16_t out_pack;
          simd16_t qv(0);
          if (valid_head) {
            const T* head_row = raw_row + static_cast<int64_t>(h) * 16;
            const float denom = denom_buf_[h];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
              const int gk = h * 16 + k;
              const float normed = static_cast<float>(head_row[k]) * denom;
              const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
              float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
              qv[k] = quant_convert<int8_t>(q_f);
            }
          }
#pragma unroll
          for (int p = 0; p < 8; ++p) {
            out_pack[p] =
                static_cast<uint16_t>(static_cast<uint8_t>(qv[p * 2])) |
                (static_cast<uint16_t>(static_cast<uint8_t>(qv[p * 2 + 1])) << 8);
          }
          out16_t* out_ptr = reinterpret_cast<out16_t*>(out_row + static_cast<int64_t>(h_base) * 16);
          if (full_chunk) {
            sycl::ext::oneapi::experimental::group_store(sg, out_pack, out_ptr);
          } else if (valid_head) {
            out_ptr[sg_lane] = out_pack;
          }
        } else {
          if (valid_head) {
            const T* head_row = raw_row + static_cast<int64_t>(h) * 16;
            const float denom = denom_buf_[h];
            out_t* out_ptr = out_row + static_cast<int64_t>(h) * 16;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
              const int gk = h * 16 + k;
              const float normed = static_cast<float>(head_row[k]) * denom;
              const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
              float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
              out_ptr[k] = quant_convert<out_t>(q_f);
            }
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    denom_buf_ = sycl_local_acc_t<float>(static_cast<size_t>(head_num_), cgh);
    const size_t wb_size = static_cast<size_t>(head_num_ > TPB ? head_num_ : TPB);
    warp_buf_ = sycl_local_acc_t<float>(wb_size, cgh);
    // SLM复用优化：按block分配，避免浪费
    raw_buf_ = sycl_local_acc_t<T>(static_cast<size_t>(head_num_) * static_cast<size_t>(HD), cgh);
  }

  HeadRMSNormDynamicQuantSmallSpecializedFunctor(
      int num_tokens, int head_num,
      float epsilon, float max_dtype_val,
      const T* X, const float* norm_weight, const float* smooth_scale,
      out_t* quant_out, float* per_token_scale)
      : num_tokens_(num_tokens), head_num_(head_num),
        epsilon_(epsilon), max_dtype_val_(max_dtype_val),
        X_(X), norm_weight_(norm_weight), smooth_scale_(smooth_scale),
        quant_out_(quant_out), per_token_scale_(per_token_scale) {}

 private:
  int   num_tokens_, head_num_;
  float epsilon_, max_dtype_val_;
  const T*     X_;
  const float* norm_weight_;
  const float* smooth_scale_;
  out_t*       quant_out_;
  float*       per_token_scale_;
  sycl_local_acc_t<float> denom_buf_;
  sycl_local_acc_t<float> warp_buf_;
  sycl_local_acc_t<T> raw_buf_;
};

// ---------------------------------------------------------------------------
// Generic small kernel fallback (keeps existing robust behavior).
// ---------------------------------------------------------------------------
template <typename T, typename out_t, int TPB, int UNROLL>
struct HeadRMSNormDynamicQuantSmallFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int NWARPS = TPB / SIMD;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int token_idx = static_cast<int>(item_id.get_group(0));
    const int tid       = static_cast<int>(item_id.get_local_id(0));

    if (token_idx >= num_tokens_) return;

    const int N_total  = head_num_ * head_dim_;
    const T*   row     = X_        + static_cast<int64_t>(token_idx) * N_total;
    out_t*     out_row = quant_out_ + static_cast<int64_t>(token_idx) * N_total;

    auto sg = item_id.get_sub_group();
    const int sg_lane = static_cast<int>(sg.get_local_linear_id());
    const int wid     = tid / SIMD;

    for (int h = wid; h < head_num_; h += NWARPS) {
      const T* head_row = row + static_cast<int64_t>(h) * head_dim_;

      float var_sum = 0.f;
      for (int k = sg_lane; k < head_dim_; k += SIMD) {
        const float v = static_cast<float>(head_row[k]);
        var_sum += v * v;
      }
      var_sum = sycl::reduce_over_group(sg, var_sum, sycl::plus<float>());

      const float denom = sycl::rsqrt(var_sum / static_cast<float>(head_dim_) + epsilon_);

      float head_abs_max = 0.f;
      for (int k = sg_lane; k < head_dim_; k += SIMD) {
        const int gk = h * head_dim_ + k;
        const float normed = static_cast<float>(head_row[k]) * denom;
        const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
        head_abs_max = sycl::fmax(head_abs_max, sycl::fabs(smoothed));
      }
      head_abs_max = sycl::reduce_over_group(sg, head_abs_max, sycl::maximum<float>());

      if (sg_lane == 0) {
        denom_buf_[h] = denom;
        warp_buf_[h] = head_abs_max;
      }
    }
    sycl::group_barrier(item_id.get_group());

    float token_abs_max = 0.f;
    for (int h = tid; h < head_num_; h += TPB)
      token_abs_max = sycl::fmax(token_abs_max, warp_buf_[h]);

    warp_buf_[tid] = token_abs_max;
    sycl::group_barrier(item_id.get_group());
    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        warp_buf_[tid] = sycl::fmax(warp_buf_[tid], warp_buf_[tid + stride]);
      sycl::group_barrier(item_id.get_group());
    }
    const float abs_max = warp_buf_[0];

    if (tid == 0)
      per_token_scale_[token_idx] = abs_max / max_dtype_val_;

    const float inv_scale = (abs_max > 0.f) ? (max_dtype_val_ / abs_max) : 0.f;
    for (int h = wid; h < head_num_; h += NWARPS) {
      const T* head_row = row + static_cast<int64_t>(h) * head_dim_;
      out_t* head_out = out_row + static_cast<int64_t>(h) * head_dim_;
      const float denom = denom_buf_[h];

      for (int k = sg_lane; k < head_dim_; k += SIMD) {
        const int gk = h * head_dim_ + k;
        const float normed = static_cast<float>(head_row[k]) * denom;
        const float smoothed = normed * norm_weight_[k] * smooth_scale_[gk];
        float q_f = sycl::clamp(smoothed * inv_scale, -max_dtype_val_, max_dtype_val_);
        head_out[k] = quant_convert<out_t>(q_f);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    denom_buf_ = sycl_local_acc_t<float>(static_cast<size_t>(head_num_), cgh);
    const size_t wb_size = static_cast<size_t>(head_num_ > TPB ? head_num_ : TPB);
    warp_buf_ = sycl_local_acc_t<float>(wb_size, cgh);
  }

  HeadRMSNormDynamicQuantSmallFunctor(
      int num_tokens, int head_num, int head_dim,
      float epsilon, float max_dtype_val,
      const T* X, const float* norm_weight, const float* smooth_scale,
      out_t* quant_out, float* per_token_scale)
      : num_tokens_(num_tokens), head_num_(head_num), head_dim_(head_dim),
        epsilon_(epsilon), max_dtype_val_(max_dtype_val),
        X_(X), norm_weight_(norm_weight), smooth_scale_(smooth_scale),
        quant_out_(quant_out), per_token_scale_(per_token_scale) {}

 private:
  int   num_tokens_, head_num_, head_dim_;
  float epsilon_, max_dtype_val_;
  const T*     X_;
  const float* norm_weight_;
  const float* smooth_scale_;
  out_t*       quant_out_;
  float*       per_token_scale_;
  sycl_local_acc_t<float> denom_buf_;
  sycl_local_acc_t<float> warp_buf_;
};

template <typename T, typename out_t>
void launch_kernel(
    int num_tokens, int head_num, int head_dim,
    float eps, float max_dtype_val,
    const T* X, const float* norm_weight, const float* smooth_scale,
  out_t* quant_out, float* per_token_scale) {
  constexpr int TPB = 256;
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  
  if constexpr (sizeof(T) > 4) {
    TORCH_CHECK(false, "Only 16-bit or 32-bit types are supported.");
  } else {
    if (head_dim < 32) {
      // ── Small path: head_dim < 32 ──
      if constexpr (sizeof(T) == 2) {
        if (head_dim == 8) {
          using KF = HeadRMSNormDynamicQuantSmallSpecializedFunctor<T, out_t, TPB, 8>;
          KF kfn(num_tokens, head_num, eps, max_dtype_val,
                 X, norm_weight, smooth_scale, quant_out, per_token_scale);
          sycl::range<1> local_r(TPB);
          sycl::range<1> global_r(static_cast<size_t>(num_tokens) * TPB);
          queue.submit([&](sycl::handler& cgh) {
            kfn.sycl_ker_config_convention(cgh);
            cgh.parallel_for(sycl::nd_range<1>(global_r, local_r), kfn);
          });
          return;
        }
        if (head_dim == 16) {
          using KF = HeadRMSNormDynamicQuantSmallSpecializedFunctor<T, out_t, TPB, 16>;
          KF kfn(num_tokens, head_num, eps, max_dtype_val,
                 X, norm_weight, smooth_scale, quant_out, per_token_scale);
          sycl::range<1> local_r(TPB);
          sycl::range<1> global_r(static_cast<size_t>(num_tokens) * TPB);
          queue.submit([&](sycl::handler& cgh) {
            kfn.sycl_ker_config_convention(cgh);
            cgh.parallel_for(sycl::nd_range<1>(global_r, local_r), kfn);
          });
          return;
        }
      }

      const int N_total = head_num * head_dim;
      const int unroll_factor = (N_total + TPB - 1) / TPB;

      auto launch_small = [&](auto unroll_tag) {
        constexpr int UNROLL_SEL = decltype(unroll_tag)::value;
        using KF = HeadRMSNormDynamicQuantSmallFunctor<T, out_t, TPB, UNROLL_SEL>;
        KF kfn(num_tokens, head_num, head_dim, eps, max_dtype_val,
               X, norm_weight, smooth_scale, quant_out, per_token_scale);
        sycl::range<1> local_r(TPB);
        sycl::range<1> global_r(static_cast<size_t>(num_tokens) * TPB);
        queue.submit([&](sycl::handler& cgh) {
          kfn.sycl_ker_config_convention(cgh);
          cgh.parallel_for(sycl::nd_range<1>(global_r, local_r), kfn);
        });
      };

      if (unroll_factor <= 1) { launch_small(std::integral_constant<int, 1>{}); }
      else if (unroll_factor <= 2) { launch_small(std::integral_constant<int, 2>{}); }
      else if (unroll_factor <= 3) { launch_small(std::integral_constant<int, 3>{}); }
      else if (unroll_factor <= 4) { launch_small(std::integral_constant<int, 4>{}); }
      else if (unroll_factor <= 8) { launch_small(std::integral_constant<int, 8>{}); }
      else if (unroll_factor <= 12) { launch_small(std::integral_constant<int, 12>{}); }
      else if (unroll_factor <= 16) { launch_small(std::integral_constant<int, 16>{}); }
      else if (unroll_factor <= 24) { launch_small(std::integral_constant<int, 24>{}); }
      else if (unroll_factor <= 32) { launch_small(std::integral_constant<int, 32>{}); }
      else if (unroll_factor <= 48) { launch_small(std::integral_constant<int, 48>{}); }
      else {
        TORCH_CHECK(false, "Hidden size (head_num * head_dim) is too large for Small kernel.");
      }
    } else {
      // ── Medium/Big path: head_dim >= 32 ──
      auto launch_medium = [&](int unroll_m, auto bpl_tag) {
        constexpr int BPL_SEL = decltype(bpl_tag)::value;
        auto launch_one = [&](auto unroll_tag) {
          constexpr int UNROLL_SEL = decltype(unroll_tag)::value;
          using KF = HeadRMSNormDynamicQuantMediumFunctor<T, out_t, TPB, UNROLL_SEL, BPL_SEL>;
          KF kfn(num_tokens, head_num, head_dim, eps, max_dtype_val,
                 X, norm_weight, smooth_scale, quant_out, per_token_scale);
          sycl::range<1> local_r(TPB);
          sycl::range<1> global_r(static_cast<size_t>(num_tokens) * TPB);
          queue.submit([&](sycl::handler& cgh) {
            kfn.sycl_ker_config_convention(cgh);
            cgh.parallel_for(sycl::nd_range<1>(global_r, local_r), kfn);
          });
        };

        if      (unroll_m <= 1)  { launch_one(std::integral_constant<int, 1>{}); }
        else if (unroll_m <= 2)  { launch_one(std::integral_constant<int, 2>{}); }
        else if (unroll_m <= 3)  { launch_one(std::integral_constant<int, 3>{}); }
        else if (unroll_m <= 4)  { launch_one(std::integral_constant<int, 4>{}); }
        else if (unroll_m <= 6)  { launch_one(std::integral_constant<int, 6>{}); }
        else if (unroll_m <= 8)  { launch_one(std::integral_constant<int, 8>{}); }
        else if (unroll_m <= 12) { launch_one(std::integral_constant<int, 12>{}); }
        else if (unroll_m <= 16) { launch_one(std::integral_constant<int, 16>{}); }
        else {
          TORCH_CHECK(false, "head_dim too large for Medium kernel.");
        }
      };

      auto launch_big = [&](auto bpl_tag) {
        constexpr int BPL_SEL = decltype(bpl_tag)::value;
        using KF = HeadRMSNormDynamicQuantBigFunctor<T, out_t, TPB, BPL_SEL>;
        KF kfn(num_tokens, head_num, head_dim, eps, max_dtype_val,
               X, norm_weight, smooth_scale, quant_out, per_token_scale);
        sycl::range<1> local_r(TPB);
        sycl::range<1> global_r(static_cast<size_t>(num_tokens) * TPB);
        queue.submit([&](sycl::handler& cgh) {
          kfn.sycl_ker_config_convention(cgh);
          cgh.parallel_for(sycl::nd_range<1>(global_r, local_r), kfn);
        });
      };

      constexpr int EPL_16 = 16 / static_cast<int>(sizeof(T));
      constexpr int EPL_8  = 8  / static_cast<int>(sizeof(T));
      constexpr int EPL_4  = (4 / static_cast<int>(sizeof(T)) > 0)
                               ? (4 / static_cast<int>(sizeof(T))) : 1;
      constexpr int STEP_16 = SIMD * EPL_16;
      constexpr int STEP_8  = SIMD * EPL_8;
      constexpr int STEP_4  = SIMD * EPL_4;

      if (head_dim % STEP_16 == 0) {
        const int unroll_m = (head_dim + STEP_16 - 1) / STEP_16;
        if (unroll_m <= 16) {
          launch_medium(unroll_m, std::integral_constant<int, 16>{});
        } else {
          launch_big(std::integral_constant<int, 16>{});
        }
      } else if (head_dim % STEP_8 == 0) {
        const int unroll_m = (head_dim + STEP_8 - 1) / STEP_8;
        if (unroll_m <= 16) {
          launch_medium(unroll_m, std::integral_constant<int, 8>{});
        } else {
          launch_big(std::integral_constant<int, 8>{});
        }
      } else if (head_dim % STEP_4 == 0) {
        const int unroll_m = (head_dim + STEP_4 - 1) / STEP_4;
        if (unroll_m <= 16) {
          launch_medium(unroll_m, std::integral_constant<int, 4>{});
        } else {
          launch_big(std::integral_constant<int, 4>{});
        }
      } else {
        TORCH_CHECK(false, "head_dim must be a multiple of 4 bytes for Medium/Big kernel vectorized load. Try padding head_dim.");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Python-facing entry point
// Inputs:
//   token_data   [num_tokens, head_num, head_dim]  bfloat16 (or fp16/fp32)
//   norm_weight  [head_dim]                         float32
//   smooth_scale [head_num * head_dim]              float32
// Outputs:
//   quant_tokens    [num_tokens, head_num * head_dim]  int8
//   per_token_scale [num_tokens]                        float32
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> head_rms_norm_dynamic_quant_forward(
    const torch::Tensor& token_data,
    const torch::Tensor& norm_weight,
    const torch::Tensor& smooth_scale,
  double eps,
  at::ScalarType out_dtype) {
  TORCH_CHECK(token_data.is_xpu(),    "token_data must be an XPU tensor");
  TORCH_CHECK(norm_weight.is_xpu(),   "norm_weight must be an XPU tensor");
  TORCH_CHECK(smooth_scale.is_xpu(),  "smooth_scale must be an XPU tensor");
  TORCH_CHECK(token_data.dim() == 3,  "token_data must be 3D [num_tokens, head_num, head_dim]");
  TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be 1D [head_dim]");
  TORCH_CHECK(smooth_scale.dim() == 1,"smooth_scale must be 1D [head_num * head_dim]");
  TORCH_CHECK(norm_weight.scalar_type()  == at::ScalarType::Float,
              "norm_weight must be float32");
  TORCH_CHECK(smooth_scale.scalar_type() == at::ScalarType::Float,
              "smooth_scale must be float32");

  const auto td = token_data.contiguous();
  const auto nw = norm_weight.contiguous();
  const auto ss = smooth_scale.contiguous();

  const int num_tokens = static_cast<int>(td.size(0));
  const int head_num   = static_cast<int>(td.size(1));
  const int head_dim   = static_cast<int>(td.size(2));

  TORCH_CHECK(nw.size(0) == head_dim,            "norm_weight size must equal head_dim");
  TORCH_CHECK(ss.size(0) == head_num * head_dim, "smooth_scale size must equal head_num * head_dim");

  float max_dtype_val = 127.0f;
  if (out_dtype == at::ScalarType::Char) {
    max_dtype_val = 127.0f;
  } else if (out_dtype == at::ScalarType::Float8_e4m3fn) {
    max_dtype_val = 448.0f;
  } else {
    TORCH_CHECK(false, "Unsupported out_dtype. Use torch.int8, or torch.float8_e4m3fn.");
  }

  auto quant_tokens = torch::empty(
      {num_tokens, head_num * head_dim},
      td.options().dtype(out_dtype));
  auto per_token_scale = torch::empty(
      {num_tokens},
      td.options().dtype(at::ScalarType::Float));

  if (num_tokens == 0) return {quant_tokens, per_token_scale};

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      td.scalar_type(), "head_rms_norm_dq_sycl", [&]() {
        if (out_dtype == at::ScalarType::Char) {
          launch_kernel<scalar_t, int8_t>(
              num_tokens, head_num, head_dim,
              static_cast<float>(eps), max_dtype_val,
              td.const_data_ptr<scalar_t>(),
              nw.data_ptr<float>(),
              ss.data_ptr<float>(),
              quant_tokens.data_ptr<int8_t>(),
              per_token_scale.data_ptr<float>());
        } else if (out_dtype == at::ScalarType::Float8_e4m3fn) {
          launch_kernel<scalar_t, c10::Float8_e4m3fn>(
              num_tokens, head_num, head_dim,
              static_cast<float>(eps), max_dtype_val,
              td.const_data_ptr<scalar_t>(),
              nw.data_ptr<float>(),
              ss.data_ptr<float>(),
              quant_tokens.data_ptr<c10::Float8_e4m3fn>(),
              per_token_scale.data_ptr<float>());
        }
      });

  return {quant_tokens, per_token_scale};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_rms_norm_dynamic_quant_forward",
        &head_rms_norm_dynamic_quant_forward,
  py::arg("token_data"),
  py::arg("norm_weight"),
  py::arg("smooth_scale"),
  py::arg("eps"),
  py::arg("out_dtype") = at::ScalarType::Char,
  "Head RMSNorm + Dynamic Quant forward (SYCL, int8/fp8_e4m3fn output)");
}
