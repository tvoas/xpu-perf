/*
 * quant_matmul_onednn.cpp
 *
 * Pybind11 module exposing oneDNN int8 matmul as a Python callable that
 * accepts PyTorch XPU tensors directly (zero-copy via SYCL interop).
 *
 * API:
 *   quant_matmul(hidden_states, per_token_scale, weight, weight_scale, out)
 *     hidden_states  : [M, K] int8,     XPU
 *     per_token_scale: [M]    float32,  XPU
 *     weight         : [N, K] int8,     XPU  (row-major, NOT transposed)
 *     weight_scale   : [N]    float32,  XPU
 *     out            : [M, N] bfloat16, XPU
 *
 * Dependencies: libdnnl.so, PyTorch XPU, oneAPI SYCL runtime.
 * Build: see build.sh (requires icpx).
 */

#include <torch/extension.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl.h>          // C API for set_scales
#include <oneapi/dnnl/dnnl_sycl.hpp>   // SYCL interop

#include <mutex>
#include <unordered_map>

using namespace dnnl;

// ───────────────────────────────────────────────────────────────────────
// Primitive cache (engine per device, primitive per shape)
// ───────────────────────────────────────────────────────────────────────

struct ShapeKey {
    int64_t M, K, N;
    bool operator==(const ShapeKey& o) const {
        return M == o.M && K == o.K && N == o.N;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const {
        size_t h = std::hash<int64_t>()(k.M);
        h ^= std::hash<int64_t>()(k.K) << 16;
        h ^= std::hash<int64_t>()(k.N) << 32;
        return h;
    }
};

struct DeviceState {
    engine eng;
    std::unordered_map<ShapeKey, matmul, ShapeKeyHash> prim_cache;
};

static std::unordered_map<int, DeviceState> g_device_states;
static std::mutex g_mutex;

static DeviceState& get_device_state(int dev_idx, sycl::queue& q) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_device_states.find(dev_idx);
    if (it != g_device_states.end()) return it->second;

    auto eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    auto [pos, _] = g_device_states.emplace(
        dev_idx, DeviceState{std::move(eng), {}});
    return pos->second;
}

static matmul& get_or_create_prim(DeviceState& state,
                                  int64_t M, int64_t K, int64_t N) {
    ShapeKey key{M, K, N};
    auto it = state.prim_cache.find(key);
    if (it != state.prim_cache.end()) return it->second;

    memory::desc src_md({M, K}, memory::data_type::s8,   memory::format_tag::ab);
    memory::desc wei_md({K, N}, memory::data_type::s8,   memory::format_tag::ba);
    memory::desc dst_md({M, N}, memory::data_type::bf16, memory::format_tag::ab);

    primitive_attr attr;
    // Per-token src scales: mask=3 (all dims), groups={1, K}
    // → group_size 1 along M (per-row), K along K (one group)
    // → M total scale values.  Matches benchdnn: src:per_tensor:f32:1xK
    dnnl_dim_t src_groups[] = {1, K};
    dnnl_primitive_attr_set_scales(
        attr.get(), DNNL_ARG_SRC, /*mask=*/3,
        /*ndims=*/2, src_groups, dnnl_f32);
    // Per-channel wei scales: mask=3 (all dims), groups={K, 1}
    // → group_size K along K (one group), 1 along N (per-col)
    // → N total scale values.  Matches benchdnn: wei:per_tensor:f32:Kx1
    dnnl_dim_t wei_groups[] = {K, 1};
    dnnl_primitive_attr_set_scales(
        attr.get(), DNNL_ARG_WEIGHTS, /*mask=*/3,
        /*ndims=*/2, wei_groups, dnnl_f32);

    auto pd   = matmul::primitive_desc(state.eng, src_md, wei_md, dst_md, attr);
    auto prim = matmul(pd);

    auto [pos, _] = state.prim_cache.emplace(key, std::move(prim));
    return pos->second;
}

// ───────────────────────────────────────────────────────────────────────
// Main API: accepts PyTorch XPU tensors, zero-copy
// ───────────────────────────────────────────────────────────────────────

static void quant_matmul(
    at::Tensor hidden_states,    // [M, K] int8
    at::Tensor per_token_scale,  // [M]    float32
    at::Tensor weight,           // [N, K] int8
    at::Tensor weight_scale,     // [N]    float32
    at::Tensor out               // [M, N] bfloat16
) {
    TORCH_CHECK(hidden_states.is_xpu(), "hidden_states must be on XPU");
    TORCH_CHECK(weight.is_xpu(),        "weight must be on XPU");
    TORCH_CHECK(out.is_xpu(),           "out must be on XPU");

    int dev_idx = hidden_states.device().index();
    auto xpu_stream = at::xpu::getCurrentXPUStream(dev_idx);
    sycl::queue& queue = xpu_stream.queue();

    auto& state = get_device_state(dev_idx, queue);
    auto  strm  = sycl_interop::make_stream(state.eng, queue);

    int64_t M = hidden_states.size(0);
    int64_t K = hidden_states.size(1);
    int64_t N = weight.size(0);

    auto& prim = get_or_create_prim(state, M, K, N);

    // Memory descriptors
    memory::desc src_md({M, K}, memory::data_type::s8,   memory::format_tag::ab);
    memory::desc wei_md({K, N}, memory::data_type::s8,   memory::format_tag::ba);
    memory::desc dst_md({M, N}, memory::data_type::bf16, memory::format_tag::ab);
    memory::desc src_scale_md({M, 1}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc wei_scale_md({1, N}, memory::data_type::f32, memory::format_tag::ab);

    // Wrap PyTorch tensor pointers as oneDNN memory (zero-copy via USM)
    auto src_mem = sycl_interop::make_memory(
        src_md, state.eng, sycl_interop::memory_kind::usm,
        hidden_states.data_ptr());
    auto wei_mem = sycl_interop::make_memory(
        wei_md, state.eng, sycl_interop::memory_kind::usm,
        weight.data_ptr());
    auto dst_mem = sycl_interop::make_memory(
        dst_md, state.eng, sycl_interop::memory_kind::usm,
        out.data_ptr());
    auto src_scale_mem = sycl_interop::make_memory(
        src_scale_md, state.eng, sycl_interop::memory_kind::usm,
        per_token_scale.data_ptr());
    auto wei_scale_mem = sycl_interop::make_memory(
        wei_scale_md, state.eng, sycl_interop::memory_kind::usm,
        weight_scale.data_ptr());

    prim.execute(strm, {
        {DNNL_ARG_SRC,                             src_mem},
        {DNNL_ARG_WEIGHTS,                         wei_mem},
        {DNNL_ARG_DST,                             dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,     src_scale_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_mem},
    });
    // No strm.wait() — PyTorch stream sync handles it
}

// ───────────────────────────────────────────────────────────────────────
// Python module
// ───────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(quant_matmul_onednn, m) {
    m.doc() = "oneDNN int8 quant_matmul with PyTorch XPU tensor interop";
    m.def("quant_matmul", &quant_matmul,
          "Run int8×int8→bf16 matmul with per-token/per-channel scales.\n\n"
          "Args:\n"
          "  hidden_states  : [M, K] int8 XPU\n"
          "  per_token_scale: [M] float32 XPU\n"
          "  weight         : [N, K] int8 XPU\n"
          "  weight_scale   : [N] float32 XPU\n"
          "  out            : [M, N] bfloat16 XPU",
          py::arg("hidden_states"),
          py::arg("per_token_scale"),
          py::arg("weight"),
          py::arg("weight_scale"),
          py::arg("out"));
}
