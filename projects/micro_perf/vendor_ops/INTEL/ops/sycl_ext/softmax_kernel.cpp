#include <torch/extension.h>
#include <ATen/Functions.h>
#include <ATen/ops/softmax.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace py = pybind11;

namespace {

inline sycl::queue& get_current_queue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

inline bool is_torch_fallback_enabled() {
  const char* env = std::getenv("SOFTMAX_FORCE_TORCH_FALLBACK");
  if (env == nullptr) {
    return true;
  }

  std::string v(env);
  for (char& c : v) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  if (v == "0" || v == "false" || v == "off" || v == "no") {
    return false;
  }
  if (v == "1" || v == "true" || v == "on" || v == "yes") {
    return true;
  }

  return true;
}

inline bool should_force_torch_fallback(
    torch::ScalarType /*st*/,
    int64_t /*batch_size*/,
    int64_t /*dim_size*/) {
  // After vec4 packed I/O, the SYCL kernel meets-or-beats torch on every
  // measured shape across fp16/bf16/fp32, so no shape-specific fallback
  // is needed. Kept as a hook for future per-shape fallback overrides.
  return false;
}

template <typename scalar_t>
inline float to_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
inline scalar_t from_float(float x) {
  return static_cast<scalar_t>(x);
}

// Packed vector POD for coalesced/wider loads & stores. For 16-bit dtypes this
// turns 4 scalar stores (each promoted to a 32-bit message on Intel GPUs) into
// a single 8-byte store; for fp32 it becomes a 16-byte store.
template <typename T, int N>
struct alignas(N * sizeof(T)) vec_t {
  T val[N];
};

template <typename scalar_t>
int pick_wg_size(int dim_size, int wg_override, int device_wg_max) {
  int wg_size = wg_override > 0 ? wg_override : 256;

  if (wg_override == 0) {
    if constexpr (std::is_same_v<scalar_t, float>) {
      if (dim_size <= 2048) {
        wg_size = 128;
      } else if (dim_size <= 8192) {
        wg_size = 256;
      } else if (dim_size <= 16384) {
        wg_size = 512;
      } else {
        wg_size = 1024;
      }
    } else if constexpr (
        std::is_same_v<scalar_t, sycl::half> ||
        std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
      if (dim_size <= 4096) {
        wg_size = 128;
      } else if (dim_size <= 16384) {
        wg_size = 256;
      } else if (dim_size <= 32768) {
        wg_size = 512;
      } else {
        wg_size = 1024;
      }
    }
  }

  if (wg_size > device_wg_max) {
    wg_size = 1;
    while (wg_size * 2 <= device_wg_max) {
      wg_size *= 2;
    }
  }

  if (dim_size < wg_size) {
    wg_size = 1;
    while (wg_size * 2 <= dim_size) {
      wg_size *= 2;
    }
  }

  return wg_size;
}

template <typename scalar_t>
void softmax_compute_into_impl(
    torch::Tensor input,
    torch::Tensor output,
    float softmax_scale,
    int64_t /*k_block_i64*/,
    int64_t wg_size_i64,
    const std::string& smalldim_mode,
    int64_t /*rows_per_group_i64*/) {
  if (input.dim() != 2 || output.dim() != 2) {
    throw std::runtime_error("input/output must be 2D [batch, dim]");
  }
  if (input.size(0) != output.size(0) || input.size(1) != output.size(1)) {
    throw std::runtime_error("input/output shape mismatch");
  }
  if (!input.is_contiguous() || !output.is_contiguous()) {
    throw std::runtime_error("input/output must be contiguous");
  }
  if (smalldim_mode != "baseline") {
    throw std::runtime_error("baseline softmax kernel only supports smalldim_mode=baseline");
  }

  const int batch = static_cast<int>(input.size(0));
  const int dim_size = static_cast<int>(input.size(1));
  if (batch <= 0 || dim_size <= 0) {
    throw std::runtime_error("batch/dim_size must be positive");
  }

  auto* d_in = reinterpret_cast<scalar_t*>(input.data_ptr());
  auto* d_out = reinterpret_cast<scalar_t*>(output.data_ptr());

  sycl::queue& q = get_current_queue();
  const int device_wg_max = static_cast<int>(
      q.get_device().get_info<sycl::info::device::max_work_group_size>());

  int wg_override = static_cast<int>(wg_size_i64);
  if (wg_override < 0) {
    throw std::runtime_error("wg_size must be >= 0 (0 means auto)");
  }
  if (wg_override > 0 && (wg_override & (wg_override - 1)) != 0) {
    throw std::runtime_error("wg_size must be power-of-two when set");
  }

  const int wg_size = pick_wg_size<scalar_t>(dim_size, wg_override, device_wg_max);
  const float scale_log2e = softmax_scale * 1.4426950408889634f;

  sycl::range<1> local_range(static_cast<std::size_t>(wg_size));
  sycl::range<1> global_range(static_cast<std::size_t>(batch) * local_range[0]);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_max(local_range, cgh);
    sycl::local_accessor<float, 1> local_sum(local_range, cgh);

    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
      int lid = static_cast<int>(item.get_local_id(0));
      int row = static_cast<int>(item.get_group(0));
      std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);

      float thread_max = -std::numeric_limits<float>::infinity();
      float thread_sum = 0.0f;

      constexpr int VEC = 4;
      constexpr int VEC_BYTES = VEC * sizeof(scalar_t);
      using vec_io = vec_t<scalar_t, VEC>;
      const scalar_t* row_in = d_in + base;
      scalar_t* row_out = d_out + base;
      const bool can_vec =
          ((reinterpret_cast<std::uintptr_t>(row_in) & (VEC_BYTES - 1)) == 0) &&
          ((reinterpret_cast<std::uintptr_t>(row_out) & (VEC_BYTES - 1)) == 0) &&
          ((dim_size & (VEC - 1)) == 0) &&
          // Require every wg lane to own >=1 vec, otherwise idle subgroups
          // would inject NaN into the online-softmax reduction
          // ((-inf)-(-inf) inside exp2()).
          (dim_size >= VEC * wg_size);

      if (can_vec) {
        const int n_vec = dim_size / VEC;
        auto* v_in = reinterpret_cast<const vec_io*>(row_in);
        for (int vi = lid; vi < n_vec; vi += wg_size) {
          vec_io tmp = v_in[vi];
          #pragma unroll
          for (int j = 0; j < VEC; ++j) {
            float v = scale_log2e * to_float(tmp.val[j]);
            float new_max = sycl::fmax(thread_max, v);
            thread_sum = thread_sum * sycl::native::exp2(thread_max - new_max) +
                         sycl::native::exp2(v - new_max);
            thread_max = new_max;
          }
        }
      } else {
        int step = wg_size * 2;
        int i = lid;
        for (; i + wg_size < dim_size; i += step) {
          float v0 = scale_log2e * to_float(d_in[base + static_cast<std::size_t>(i)]);
          float new_max0 = sycl::fmax(thread_max, v0);
          thread_sum = thread_sum * sycl::native::exp2(thread_max - new_max0) +
                       sycl::native::exp2(v0 - new_max0);
          thread_max = new_max0;

          int i1 = i + wg_size;
          float v1 = scale_log2e * to_float(d_in[base + static_cast<std::size_t>(i1)]);
          float new_max1 = sycl::fmax(thread_max, v1);
          thread_sum = thread_sum * sycl::native::exp2(thread_max - new_max1) +
                       sycl::native::exp2(v1 - new_max1);
          thread_max = new_max1;
        }
        for (; i < dim_size; i += wg_size) {
          float v = scale_log2e * to_float(d_in[base + static_cast<std::size_t>(i)]);
          float new_max = sycl::fmax(thread_max, v);
          thread_sum = thread_sum * sycl::native::exp2(thread_max - new_max) +
                       sycl::native::exp2(v - new_max);
          thread_max = new_max;
        }
      }

      if (wg_size >= 512) {
        local_max[lid] = thread_max;
        local_sum[lid] = thread_sum;
        item.barrier(sycl::access::fence_space::local_space);

        for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
          if (lid < stride) {
            float m0 = local_max[lid];
            float s0 = local_sum[lid];
            float m1 = local_max[lid + stride];
            float s1 = local_sum[lid + stride];
            float merged_max = sycl::fmax(m0, m1);
            float merged_sum = s0 * sycl::native::exp2(m0 - merged_max) +
                               s1 * sycl::native::exp2(m1 - merged_max);
            local_max[lid] = merged_max;
            local_sum[lid] = merged_sum;
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
      } else {
        sycl::sub_group sg = item.get_sub_group();
        int sg_size = static_cast<int>(sg.get_local_range().size());
        int sg_id = lid / sg_size;
        int num_sg = (wg_size + sg_size - 1) / sg_size;

        float sg_max = sycl::reduce_over_group(sg, thread_max, sycl::maximum<float>());
        float sg_sum = sycl::reduce_over_group(
            sg,
            thread_sum * sycl::native::exp2(thread_max - sg_max),
            sycl::plus<float>());

        if (sg.leader()) {
          local_max[sg_id] = sg_max;
          local_sum[sg_id] = sg_sum;
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (lid == 0) {
          float row_max = local_max[0];
          float row_sum = local_sum[0];
          for (int g = 1; g < num_sg; ++g) {
            float m1 = local_max[g];
            float s1 = local_sum[g];
            float merged_max = sycl::fmax(row_max, m1);
            float merged_sum = row_sum * sycl::native::exp2(row_max - merged_max) +
                               s1 * sycl::native::exp2(m1 - merged_max);
            row_max = merged_max;
            row_sum = merged_sum;
          }
          local_max[0] = row_max;
          local_sum[0] = row_sum;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      float row_max = local_max[0];
      float inv_sum = 1.0f / local_sum[0];

      if (can_vec) {
        const int n_vec = dim_size / VEC;
        auto* v_in = reinterpret_cast<const vec_io*>(row_in);
        auto* v_out = reinterpret_cast<vec_io*>(row_out);
        for (int vi = lid; vi < n_vec; vi += wg_size) {
          vec_io tin = v_in[vi];
          vec_io tout;
          #pragma unroll
          for (int j = 0; j < VEC; ++j) {
            float v = scale_log2e * to_float(tin.val[j]);
            float p = sycl::native::exp2(v - row_max) * inv_sum;
            tout.val[j] = from_float<scalar_t>(p);
          }
          v_out[vi] = tout;
        }
      } else {
        int step_out = wg_size * 2;
        int i_out = lid;
        for (; i_out + wg_size < dim_size; i_out += step_out) {
          float v0 = scale_log2e * to_float(d_in[base + static_cast<std::size_t>(i_out)]);
          float p0 = sycl::native::exp2(v0 - row_max) * inv_sum;
          d_out[base + static_cast<std::size_t>(i_out)] = from_float<scalar_t>(p0);

          int i1 = i_out + wg_size;
          float v1 = scale_log2e * to_float(d_in[base + static_cast<std::size_t>(i1)]);
          float p1 = sycl::native::exp2(v1 - row_max) * inv_sum;
          d_out[base + static_cast<std::size_t>(i1)] = from_float<scalar_t>(p1);
        }
        for (; i_out < dim_size; i_out += wg_size) {
          float v = scale_log2e * to_float(d_in[base + static_cast<std::size_t>(i_out)]);
          float p = sycl::native::exp2(v - row_max) * inv_sum;
          d_out[base + static_cast<std::size_t>(i_out)] = from_float<scalar_t>(p);
        }
      }
    });
  });
}

void softmax_compute_into(
    torch::Tensor input,
    torch::Tensor output,
    double softmax_scale,
    int64_t k_block,
    int64_t wg_size,
    const std::string& smalldim_mode,
    int64_t rows_per_group) {
  if (input.scalar_type() != output.scalar_type()) {
    throw std::runtime_error("input/output dtype mismatch");
  }

  const auto st = input.scalar_type();
    if (is_torch_fallback_enabled() &&
      input.dim() == 2 &&
      should_force_torch_fallback(st, input.size(0), input.size(1))) {
    auto scaled_input = static_cast<float>(softmax_scale) == 1.0f
        ? input
        : input * static_cast<float>(softmax_scale);
    at::softmax_out(output, scaled_input, -1, c10::nullopt);
    return;
  }

  if (st == torch::kFloat32) {
    softmax_compute_into_impl<float>(
        input,
        output,
        static_cast<float>(softmax_scale),
        k_block,
        wg_size,
        smalldim_mode,
        rows_per_group);
    return;
  }
  if (st == torch::kFloat16) {
    softmax_compute_into_impl<sycl::half>(
        input,
        output,
        static_cast<float>(softmax_scale),
        k_block,
        wg_size,
        smalldim_mode,
        rows_per_group);
    return;
  }
  if (st == torch::kBFloat16) {
    softmax_compute_into_impl<sycl::ext::oneapi::bfloat16>(
        input,
        output,
        static_cast<float>(softmax_scale),
        k_block,
        wg_size,
        smalldim_mode,
        rows_per_group);
    return;
  }

  throw std::runtime_error("Unsupported input dtype for softmax_compute_into");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "softmax_compute_into",
      &softmax_compute_into,
      "Compute softmax into provided output tensor",
      py::arg("input"),
      py::arg("output"),
      py::arg("softmax_scale") = 1.0,
      py::arg("k_block") = 256,
      py::arg("wg_size") = 0,
      py::arg("smalldim_mode") = "baseline",
      py::arg("rows_per_group") = 1);
}
