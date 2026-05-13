import pathlib
import os
from functools import partial

import torch
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice as _libd

from xpu_perf.micro_perf.core.op import ProviderRegistry
ScaleDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["scale_dynamic_quant"]


# Triton single-pass kernel
# amax / normalize 共享同一份 fp32 buffer，DRAM 上 hidden_states 只读 1 次
INT8_MAX = 127.0
FP8_E4M3_MAX = 448.0


# torch dtype -> (kmax, triton dtype, is_integer)
_DTYPE_INFO = {
    torch.int8:          (INT8_MAX,     tl.int8,        True),
    torch.float8_e4m3fn: (FP8_E4M3_MAX, tl.float8e4nv,  False),
}


@triton.jit
def _smooth_dyn_quant_kernel(
    X_ptr, S_ptr, Q_ptr, PS_ptr,
    stride_xn, stride_qn,
    H,
    BLOCK_H: tl.constexpr,
    KMAX: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    IS_INT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(X_ptr + pid * stride_xn + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xs = x * s  # [BLOCK_H] fp32, 留在寄存器

    amax = tl.max(tl.abs(xs), axis=0)
    inv_scale = KMAX / amax
    scale = amax / KMAX

    q = xs * inv_scale
    q = tl.clamp(q, -KMAX, KMAX)
    if IS_INT:
        q = _libd.rint(q).to(OUT_DTYPE)
    else:
        q = q.to(OUT_DTYPE)

    tl.store(Q_ptr + pid * stride_qn + offs, q, mask=mask)
    tl.store(PS_ptr + pid, scale)


def _smooth_dyn_quant_triton(hidden_states, smooth_scale, out_q, out_s):
    N, H = hidden_states.shape
    BLOCK_H = triton.next_power_of_2(H)
    kmax, out_dtype, is_int = _DTYPE_INFO[out_q.dtype]
    # offline-tuned per-BLOCK_H meta-params (autotune 跑 (warps,stages) 网格得到)
    if BLOCK_H <= 1024:
        num_warps, num_stages = 8, 3
    elif BLOCK_H <= 4096:
        num_warps, num_stages = 32, 3
    else:
        num_warps, num_stages = 16, 2
    _smooth_dyn_quant_kernel[(N,)](
        hidden_states, smooth_scale, out_q, out_s,
        hidden_states.stride(0), out_q.stride(0),
        H,
        BLOCK_H=BLOCK_H,
        KMAX=kmax,
        OUT_DTYPE=out_dtype,
        IS_INT=is_int,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# 备选 torch.compile 路径，可通过 SDQ_USE_TRITON=0 切换
_COMPILED_CACHE = {}


def _make_compiled(out_dtype, kmax, is_int):
    def _fn(hidden_states, smooth_scale, out_q, out_s):
        x = hidden_states.float() * smooth_scale
        amax = x.abs().amax(dim=-1)
        scale = amax / kmax
        out_s.copy_(scale)
        q = (x / scale.unsqueeze(-1)).clamp(-kmax, kmax)
        if is_int:
            q = q.round()
        out_q.copy_(q.to(out_dtype))

    return torch.compile(_fn, dynamic=False, fullgraph=True, mode="reduce-overhead")


def _get_compiled(out_dtype):
    fn = _COMPILED_CACHE.get(out_dtype)
    if fn is None:
        kmax, _, is_int = _DTYPE_INFO[out_dtype]
        fn = _make_compiled(out_dtype, kmax, is_int)
        _COMPILED_CACHE[out_dtype] = fn
    return fn


@ProviderRegistry.register_vendor_impl("scale_dynamic_quant", "torch")
class ScaleDynamicQuantTorchOp(ScaleDynamicQuantOp):
    def vendor_parser(self):
        # torch provider supports int8 / float8 (e4m3) outputs.
        # "float8" maps to torch.float8_e4m3fn in TORCH_DTYPE_MAPPING.
        if self.dtype in ("bfloat16", "float16") \
            and self.dst_dtype in ("int8", "float8"):
            return
        raise ValueError(
            f"{type(self).__name__} only supports bfloat16/float16 -> "
            f"int8/float8, but got "
            f"dtype={self.dtype}, dst_dtype={self.dst_dtype}"
        )

    def vendor_impl(self):
        super().vendor_impl()
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )
        self._use_triton = os.environ.get("SDQ_USE_TRITON", "1") != "0"
        if not self._use_triton:
            self._compiled_fn = _get_compiled(self.dst_torch_dtype)

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]
        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        if self._use_triton:
            _smooth_dyn_quant_triton(
                hidden_states, smooth_scale, quant_tokens, per_token_scale,
            )
        else:
            self._compiled_fn(hidden_states, smooth_scale, quant_tokens, per_token_scale)
        return quant_tokens, per_token_scale
