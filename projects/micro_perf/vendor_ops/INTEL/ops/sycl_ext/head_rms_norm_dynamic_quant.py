import os
import pathlib
import importlib.util

import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.utils import calc_tensor_size
BaseHeadRMSNormDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["head_rms_norm_dynamic_quant"]

_OP_DIR = pathlib.Path(__file__).resolve().parent
_SYCL_SO = _OP_DIR / "head_rms_norm_dynamic_quant_sycl.so"

try:
    _spec = importlib.util.spec_from_file_location("head_rms_norm_dynamic_quant_sycl", str(_SYCL_SO))
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Failed to create import spec for {_SYCL_SO}")
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("head_rms_norm_dynamic_quant", "sycl_ext")
    class HeadRMSNormDynamicQuantSyclExtOp(BaseHeadRMSNormDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def vendor_impl(self):
            super().vendor_impl()
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            token_data = tensor_mapping["token_data"]
            norm_weight = tensor_mapping["norm_weight"]
            smooth_scale = tensor_mapping["smooth_scale"]

            if not norm_weight.is_contiguous():
                norm_weight = norm_weight.contiguous()
            if not smooth_scale.is_contiguous():
                smooth_scale = smooth_scale.contiguous()

            quant_tokens, per_token_scale = _sycl_ext.head_rms_norm_dynamic_quant_forward(
                token_data,
                norm_weight,
                smooth_scale,
                float(self.eps),
                self.dst_torch_dtype,
            )
            return quant_tokens, per_token_scale

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL head_rms_norm_dynamic_quant extension: {e}")
