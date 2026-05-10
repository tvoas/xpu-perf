#!/bin/bash
# Build SYCL kernels used by xpu-perf sycl_ext ops.
# Usage: source /opt/intel/oneapi/setvars.sh && bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icpx not found. Please source oneAPI setvars first."
    exit 1
fi

# Get torch include/lib paths
TORCH_INCLUDES=$(python3 -c "
import torch.utils.cpp_extension as ext
for p in ext.include_paths():
    print(f'-I{p}', end=' ')
")

TORCH_LIBS=$(python3 -c "
import torch.utils.cpp_extension as ext
for p in ext.library_paths():
    print(f'-L{p}', end=' ')
")

# Python include
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

XPU_PERF_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SYCL_TLA_ROOT="$(cd "$XPU_PERF_ROOT/../sycl-tla" && pwd)"

MKLROOT=${MKLROOT:-/opt/intel/oneapi/mkl/latest}
TBBROOT=${TBBROOT:-/opt/intel/oneapi/tbb/latest}
CMPLR_ROOT=${CMPLR_ROOT:-/opt/intel/oneapi/compiler/latest}

SYCL_TLA_INCLUDES="-I$SYCL_TLA_ROOT/include -I$SYCL_TLA_ROOT/tools/util/include -I$SYCL_TLA_ROOT/examples/common -isystem $MKLROOT/include"

SYCL_TLA_COMPILE_FLAGS="-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -DMKL_ILP64 -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen -Wall -Wno-unused-variable -Wno-unused-local-typedef -Wno-unused-but-set-variable -Wno-uninitialized -Wno-reorder-ctor -Wno-logical-op-parentheses -Wno-unused-function -Wno-unknown-pragmas"
SYCL_TLA_LINK_FLAGS="-fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen"
SYCL_TLA_LIB_DIRS="-L$MKLROOT/lib -L$TBBROOT/lib/intel64/gcc4.8"
SYCL_TLA_LINK_LIBS="$MKLROOT/lib/libmkl_intel_thread.so $CMPLR_ROOT/lib/libiomp5.so $MKLROOT/lib/libmkl_intel_ilp64.so $MKLROOT/lib/libmkl_core.so -fsycl $MKLROOT/lib/libmkl_sycl_blas.so $MKLROOT/lib/libmkl_tbb_thread.so $SYCL_TLA_LIB_DIRS -ltbb -lsycl -lOpenCL -lm -ldl -lpthread"
SYCL_TLA_RUNTIME_PATHS=(-Wl,-rpath,/lib64/stubs -Wl,-rpath,"$MKLROOT/lib" -Wl,-rpath,"$TBBROOT/lib/intel64/gcc4.8")
BMG_09_LINK_FLAGS=(-Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
BMG_10_LINK_FLAGS=(-Xs "-options \"-igc_opts 'allowDecompose2DBlockFuncs=0'\"")

VLLM_XPU_AOT_DEVICES=${VLLM_XPU_AOT_DEVICES:-pvc,bmg,bmg-g21-a0,bmg-g31-a0}
VLLM_MOE_COMPILE_FLAGS=(-fsycl -fsycl-targets=spir64_gen -fno-sycl-instrument-device-code -O3 -DNDEBUG -std=c++17)
VLLM_MOE_LINK_FLAGS=(-fsycl -fsycl-targets=spir64_gen -fsycl-max-parallel-link-jobs=16 -flink-huge-device-code -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
if [ -n "$VLLM_XPU_AOT_DEVICES" ]; then
    VLLM_MOE_LINK_FLAGS+=(-Xsycl-target-backend=spir64_gen "-device $VLLM_XPU_AOT_DEVICES")
fi

echo "Building moe_swiglu_dynamic_quant SYCL extension..."
icpx "${VLLM_MOE_COMPILE_FLAGS[@]}" -shared -fPIC \
    -DTORCH_EXTENSION_NAME=moe_swiglu_dynamic_quant_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    moe_swiglu_dynamic_quant_kernel.cpp \
    -o moe_swiglu_dynamic_quant_sycl.so \
    "${VLLM_MOE_LINK_FLAGS[@]}" \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/moe_swiglu_dynamic_quant_sycl.so"
ls -la moe_swiglu_dynamic_quant_sycl.so

echo ""
echo "Building moe_scatter_dynamic_quant SYCL extension..."
icpx "${VLLM_MOE_COMPILE_FLAGS[@]}" -shared -fPIC \
    -DTORCH_EXTENSION_NAME=moe_scatter_dynamic_quant_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    moe_scatter_dynamic_quant_kernel.cpp \
    -o moe_scatter_dynamic_quant_sycl.so \
    "${VLLM_MOE_LINK_FLAGS[@]}" \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu


echo "Built: $SCRIPT_DIR/moe_scatter_dynamic_quant_sycl.so"
ls -la moe_scatter_dynamic_quant_sycl.so
