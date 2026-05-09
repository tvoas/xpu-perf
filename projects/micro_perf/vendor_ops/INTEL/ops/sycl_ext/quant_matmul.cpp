/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel Xe W8A8 quantized matmul example with fused
           per-token (A) scale, per-channel (B) scale, and per-channel bias.

    Equivalent math to IPEX `torch.ops.torch_ipex.mm_w8a8`:

        acc[m,n] = sum_k  A[m,k] * B[k,n]                  (int32 accumulator)
        D[m,n]   = bf16( acc[m,n] * scale_a[m] * scale_b[n] + bias[n] )

    Inputs / Outputs:
        A         : int8,  [M, K] RowMajor
        B         : int8,  [K, N] RowMajor
        scale_a   : float, [M]      (per-token / per-row, broadcast along N)
        scale_b   : float, [N]      (per-channel / per-col, broadcast along M)
        bias      : float, [N]      (broadcast along M; pass null to skip)
        D         : bf16,  [M, N] RowMajor

    The fused epilogue is built as a custom EVT (Epilogue Visitor Tree):

        D = bf16(  homogeneous_multiply_add(
                       XeRowBroadcast(scale_b),
                       multiplies(AccFetch, XeColBroadcast(scale_a)),
                       XeRowBroadcast(bias)
                   )
                )
*/

#include <pybind11/pybind11.h>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Custom W8A8 fusion operation + Xe FusionCallbacks specialization
//
//   D[m,n] = bf16(  scale_b[n] * (acc[m,n] * scale_a[m]) + bias[n]  )
//
// We define a new operation tag (W8A8Op) and a corresponding FusionCallbacks
// specialization for `epilogue::IntelXeGeneric` so that the Xe collective
// epilogue can correctly extract `ThreadEpilogueOp::ElementCompute` etc.
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::epilogue::fusion {

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementScale_,
  class ElementBias_,
  int   AlignmentScale_ = 128 / cutlass::sizeof_bits<ElementScale_>::value,
  int   AlignmentBias_  = 128 / cutlass::sizeof_bits<ElementBias_>::value,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct W8A8Op : FusionOperation {
  using ElementOutput  = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScale   = ElementScale_;
  using ElementBias    = ElementBias_;
  static constexpr int AlignmentScale = AlignmentScale_;
  static constexpr int AlignmentBias  = AlignmentBias_;
  static constexpr FloatRoundStyle RoundStyle = RoundStyle_;

  static constexpr bool IsPerRowScaleSupported = true;   // scale_a[M]
  static constexpr bool IsPerColScaleSupported = true;   // scale_b[N]
  static constexpr bool IsPerColBiasSupported  = true;   // bias[N]
};

// FusionCallbacks specialization for W8A8Op on IntelXeGeneric.
// Tree shape:
//   D = bf16(  homogeneous_multiply_add(
//                  RowScaleB,
//                  multiplies(AccFetch, ColScaleA),
//                  RowBias
//              ) )
template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementScale_,
  class ElementBias_,
  int   AlignmentScale_,
  int   AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    W8A8Op<ElementOutput_, ElementCompute_, ElementScale_, ElementBias_,
           AlignmentScale_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90EVT<
      Sm90Compute<cutlass::homogeneous_multiply_add, ElementOutput_, ElementCompute_, RoundStyle_>,
      XeRowBroadcast<0, CtaTileShapeMNK_, ElementScale_, ElementCompute_, Stride<_0, _1, int64_t>, AlignmentScale_>,
      Sm90EVT<
        Sm90Compute<cutlass::multiplies, ElementCompute_, ElementCompute_, RoundStyle_>,
        Sm90AccFetch,
        XeColBroadcast<0, CtaTileShapeMNK_, ElementScale_, ElementCompute_, Stride<_1, _0, int64_t>, AlignmentScale_>
      >,
      XeRowBroadcast<0, CtaTileShapeMNK_, ElementBias_,  ElementCompute_, Stride<_0, _1, int64_t>, AlignmentBias_>
    >
{
  using ElementOutput  = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScale   = ElementScale_;
  using ElementBias    = ElementBias_;
  static constexpr int AlignmentScale = AlignmentScale_;
  static constexpr int AlignmentBias  = AlignmentBias_;

  using Impl = Sm90EVT<
      Sm90Compute<cutlass::homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle_>,
      XeRowBroadcast<0, CtaTileShapeMNK_, ElementScale, ElementCompute, Stride<_0, _1, int64_t>, AlignmentScale>,
      Sm90EVT<
        Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle_>,
        Sm90AccFetch,
        XeColBroadcast<0, CtaTileShapeMNK_, ElementScale, ElementCompute, Stride<_1, _0, int64_t>, AlignmentScale>
      >,
      XeRowBroadcast<0, CtaTileShapeMNK_, ElementBias,  ElementCompute, Stride<_0, _1, int64_t>, AlignmentBias>
  >;

  using Operation = W8A8Op<ElementOutput, ElementCompute, ElementScale, ElementBias,
                           AlignmentScale, AlignmentBias, RoundStyle_>;

  using StrideScaleA = Stride<_1, _0, int64_t>;
  using StrideScaleB = Stride<_0, _1, int64_t>;
  using StrideBias   = Stride<_0, _1, int64_t>;

  // Host-side argument struct mirrored after IPEX's mm_w8a8 ABI.
  struct Arguments {
    ElementScale const* ptr_scale_a = nullptr;   // [M*L]
    StrideScaleA        dScaleA     = {};
    ElementScale const* ptr_scale_b = nullptr;   // [N*L]
    StrideScaleB        dScaleB     = {};
    ElementBias  const* ptr_bias    = nullptr;   // [N*L] or nullptr to skip
    StrideBias          dBias       = {};

    // Convert to Impl's nested-tuple form expected by Sm90VisitorImplBase.
    // Tree (children first, NodeOp last):
    //   ( RowScaleB_args , InnerEVT_args , RowBias_args , OuterCompute_args )
    operator typename Impl::Arguments() const {
      return {
        { ptr_scale_b, ElementScale(0), dScaleB },             // leaf 0: scale_b
        {                                                       // leaf 1: inner EVT
          {},                                                   //   AccFetch (no args)
          { ptr_scale_a, ElementScale(0), dScaleA },            //   scale_a col-broadcast
          {}                                                    //   multiplies args
        },
        { ptr_bias, ElementBias(0), dBias },                   // leaf 2: bias (nullptr -> 0)
        {}                                                      // outer multiply_add args
      };
    }
  };

  using Impl::Impl;
};

} // namespace cutlass::epilogue::fusion

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help{false};
  bool error{false};
  int  m, n, k, l, iterations, verify;
  int  with_bias;     // 0 = no bias, 1 = use bias
  float scale_a_val;  // constant value to fill scale_a (so we can verify)
  float scale_b_val;  // constant value to fill scale_b
  float bias_val;     // constant value to fill bias

  Options():
    m(4096), n(4096), k(7168), l(1), iterations(20), verify(1),
    with_bias(1), scale_a_val(1.0f / 64.f), scale_b_val(1.0f / 64.f), bias_val(0.5f) { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 7168);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("verify", verify, 1);
    cmd.get_cmd_line_argument("with_bias", with_bias, 1);
    cmd.get_cmd_line_argument("scale_a", scale_a_val, 1.0f / 64.f);
    cmd.get_cmd_line_argument("scale_b", scale_b_val, 1.0f / 64.f);
    cmd.get_cmd_line_argument("bias", bias_val, 0.5f);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "W8A8 Quantized Matmul (int8 x int8 -> bf16) with fused per-token/per-channel scale + bias\n\n"
        << "Options:\n\n"
        << "  --help                      Display this usage statement\n"
        << "  --m=<int>                   M extent (default 4096)\n"
        << "  --n=<int>                   N extent (default 4096)\n"
        << "  --k=<int>                   K extent (default 7168)\n"
        << "  --l=<int>                   Batch (default 1)\n"
        << "  --with_bias=<0|1>           Whether to add per-channel bias (default 1)\n"
        << "  --scale_a=<float>           Constant value used to fill scale_a[M] (default 1/64)\n"
        << "  --scale_b=<float>           Constant value used to fill scale_b[N] (default 1/64)\n"
        << "  --bias=<float>              Constant value used to fill bias[N] (default 0.5)\n"
        << "  --iterations=<int>          Iterations (default 20)\n"
        << "  --verify=<0|1>              Verify against a reference (default 1)\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideScaleA = Stride<_1, _0, int64_t>;
  using StrideScaleB = Stride<_0, _1, int64_t>;
  using StrideBias   = Stride<_0, _1, int64_t>;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA>      block_A;
  cutlass::DeviceAllocation<ElementB>      block_B;
  cutlass::DeviceAllocation<ElementC>      block_C;        // unused (no source)
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<float>         block_scale_a;  // [M*L]
  cutlass::DeviceAllocation<float>         block_scale_b;  // [N*L]
  cutlass::DeviceAllocation<float>         block_bias;     // [N*L]

  bool with_bias = true;

  /// Reference: int32 acc on device, then host-side scale/bias/cast to bf16.
  bool verify(const ProblemShapeType& problem_size) {
    auto [M, N, K, L] = problem_size;

    cutlass::DeviceAllocation<int32_t> ref_acc(static_cast<size_t>(M) * N * L);

    cutlass::TensorRef ref_A(block_A.get(),  LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(),  LayoutB::packed({K, N}));
    cutlass::TensorRef ref_Dint(ref_acc.get(), LayoutC::packed({M, N}));

    cutlass::reference::device::GemmComplex(
        {M, N, K},
        int32_t(1),
        ref_A, cutlass::ComplexTransform::kNone,
        ref_B, cutlass::ComplexTransform::kNone,
        int32_t(0),
        ref_Dint,                  // C unused (beta=0)
        ref_Dint,                  // D = acc
        ElementAccumulator(0),
        L, M*K, K*N, M*N, M*N);
    compat::wait();

    std::vector<int32_t>       h_acc(static_cast<size_t>(M) * N * L);
    std::vector<float>         h_sa (static_cast<size_t>(M) * L);
    std::vector<float>         h_sb (static_cast<size_t>(N) * L);
    std::vector<float>         h_bs (static_cast<size_t>(N) * L);
    std::vector<ElementOutput> h_ref(static_cast<size_t>(M) * N * L);

    compat::memcpy(h_acc.data(), ref_acc.get(),       h_acc.size() * sizeof(int32_t));
    compat::memcpy(h_sa.data(),  block_scale_a.get(), h_sa.size()  * sizeof(float));
    compat::memcpy(h_sb.data(),  block_scale_b.get(), h_sb.size()  * sizeof(float));
    if (with_bias) {
      compat::memcpy(h_bs.data(), block_bias.get(), h_bs.size() * sizeof(float));
    }
    compat::wait();

    for (int b = 0; b < L; ++b) {
      for (int m = 0; m < M; ++m) {
        float sa = h_sa[b * M + m];
        for (int n = 0; n < N; ++n) {
          float sb     = h_sb[b * N + n];
          float bias_v = with_bias ? h_bs[b * N + n] : 0.f;
          float acc    = static_cast<float>(h_acc[(b * M + m) * N + n]);
          h_ref[(b * M + m) * N + n] = ElementOutput(acc * sa * sb + bias_v);
        }
      }
    }
    compat::memcpy(block_ref_D.get(), h_ref.data(), h_ref.size() * sizeof(ElementOutput));
    compat::wait();

    ElementOutput const epsilon(2e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
    return passed;
  }

  void initialize(const ProblemShapeType& problem_size, Options const& opts) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    block_scale_a.reset(static_cast<std::size_t>(M) * L);
    block_scale_b.reset(static_cast<std::size_t>(N) * L);
    block_bias.reset   (static_cast<std::size_t>(N) * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    std::vector<float> sa(static_cast<size_t>(M) * L, opts.scale_a_val);
    std::vector<float> sb(static_cast<size_t>(N) * L, opts.scale_b_val);
    std::vector<float> bs(static_cast<size_t>(N) * L, opts.with_bias ? opts.bias_val : 0.f);
    compat::memcpy(block_scale_a.get(), sa.data(), sa.size() * sizeof(float));
    compat::memcpy(block_scale_b.get(), sb.data(), sb.size() * sizeof(float));
    compat::memcpy(block_bias.get(),    bs.data(), bs.size() * sizeof(float));
    compat::wait();

    with_bias = (opts.with_bias != 0);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size, options);

    StrideScaleA dScaleA{_1{}, _0{}, int64_t(options.m)};   // batch stride = M
    StrideScaleB dScaleB{_0{}, _1{}, int64_t(options.n)};   // batch stride = N
    StrideBias   dBias  {_0{}, _1{}, int64_t(options.n)};

    // EVT arguments tree (mirror of FullEVT type defined in main):
    //   root  = homogeneous_multiply_add( scale_b , (acc * scale_a) , bias )
    typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
      {                                              // FusionCallbacks::Arguments
        block_scale_a.get(), dScaleA,
        block_scale_b.get(), dScaleB,
        (with_bias ? block_bias.get() : nullptr), dBias
      },
      block_C.get(), stride_C,
      block_D.get(), stride_D
    };

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      epilogue_args,
      hw_info
    };

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n
                << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Warm-up
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(problem_size);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      double bytes = (double(options.m) * options.k +
                      double(options.k) * options.n +
                      2.0 * double(options.m) * options.n) * options.l;
      double gbps = bytes / cute_time / 1e9;

      std::cout << "Problem Size: " << options.m << 'x' << options.n
                << 'x' << options.k << 'x' << options.l
                << "  (int8 x int8 -> bf16, +scale_a +scale_b "
                << (with_bias ? "+bias" : "no_bias") << ")\n";
      printf("Cutlass W8A8 Performance:    [%6.2f] TOp/s   %8.4f ms   %7.1f GB/s\n",
             tops / cute_time, cute_time * 1000.0, gbps);
    }
    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Templated W8A8 runner — one instantiation per (TileShape, Subgroup-Layout) config.
// We instantiate two configs and dispatch by M at runtime: large-M (tile 256x256x32, 32 SGs)
// and small-M (tile 32x256x32, 16 SGs) to avoid wasting M-tile rows when M is tiny.
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TileShape_, class SgLayout_>
cutlass::Status run_w8a8(Options const& options,
                         cutlass::KernelHardwareInfo const& hw_info,
                         char const* tag) {
  using ElementInputA          = int8_t;
  using ElementInputB          = int8_t;
  using ElementAccumulator     = int32_t;
  using ElementComputeEpilogue = float;
  using ElementOutput          = cutlass::bfloat16_t;
  using ElementScale           = float;
  using ElementBias            = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TileShape = TileShape_;
  using SgLayout  = SgLayout_;

  using TiledMma =
    typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementInputA>>,
      Layout<TileShape>,
      SgLayout
    >::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy     = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  constexpr int AlignmentScale = 128 / cutlass::sizeof_bits<ElementScale>::value;
  constexpr int AlignmentBias  = 128 / cutlass::sizeof_bits<ElementBias>::value;
  constexpr auto Round = cutlass::FloatRoundStyle::round_to_nearest;

  using EpilogueOp = cutlass::epilogue::fusion::W8A8Op<
      ElementOutput, ElementComputeEpilogue, ElementScale, ElementBias,
      AlignmentScale, AlignmentBias, Round>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape,
      decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB, cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA, void, void, cute::identity,
      GmemTiledCopyB, void, void, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  std::cout << "[config] " << tag << std::endl;
  ExampleRunner<Gemm> runner;
  return runner.run(options, hw_info);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Python entry point: same dispatch logic as the original main(), but parameters
// are passed in from Python and the function returns 0 on success.
///////////////////////////////////////////////////////////////////////////////////////////////////
int run_quant_matmul(
    int m,
    int n,
    int k,
    int l,
    int with_bias,
    float scale_a,
    float scale_b,
    float bias,
    int iterations,
    int verify) {
  Options options;
  options.m = m;
  options.n = n;
  options.k = k;
  options.l = l;
  options.with_bias = with_bias;
  options.scale_a_val = scale_a;
  options.scale_b_val = scale_b;
  options.bias_val = bias;
  options.iterations = iterations;
  options.verify = verify;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status;
  if (options.m <= 64 && options.n <= 3072) {
    status = run_w8a8<
        Shape<_64, _128, _64>,
        Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>
    >(options, hw_info, "tile <64, 128, 64>, SG <2, 4, 1>");
  } else if (options.m <= 64) {
    status = run_w8a8<
        Shape<_64, _256, _64>,
        Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>
    >(options, hw_info, "tile <64, 256, 64>, SG <2, 8, 1>");
  } else if (options.m <= 128 && options.n <= 3072) {
    status = run_w8a8<
        Shape<_128, _128, _64>,
        Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>
    >(options, hw_info, "tile <128, 128, 64>, SG <2, 8, 1>");
  } else if (options.m <= 128) {
    status = run_w8a8<
        Shape<_128, _256, _64>,
        Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>
    >(options, hw_info, "tile <128, 256, 64>, SG <4, 8, 1>");
  } else {
    status = run_w8a8<
        Shape<_256, _256, _32>,
        Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
    >(options, hw_info, "big-M tile <256, 256, 32>, SG <8, 4, 1>");
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("run_quant_matmul failed: ") +
                             cutlassGetStatusString(status));
  }
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run_quant_matmul",
      &run_quant_matmul,
      pybind11::arg("m") = 4096,
      pybind11::arg("n") = 4096,
      pybind11::arg("k") = 7168,
      pybind11::arg("l") = 1,
      pybind11::arg("with_bias") = 1,
      pybind11::arg("scale_a") = 1.0f / 64.f,
      pybind11::arg("scale_b") = 1.0f / 64.f,
      pybind11::arg("bias") = 0.5f,
      pybind11::arg("iterations") = 50,
      pybind11::arg("verify") = 0,
      "Run W8A8 quantized matmul (int8 x int8 -> bf16) with CUTLASS SYCL backend.");
}
