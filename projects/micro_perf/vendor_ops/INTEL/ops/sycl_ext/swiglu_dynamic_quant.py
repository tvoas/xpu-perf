import os
import sys
import importlib.util
import torch
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
SwigluDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["swiglu_dynamic_quant"]

# Dynamically load the localized SYCL extension
so_path = os.path.join(os.path.dirname(__file__), "swiglu_dynamic_quant_sycl.so")
if os.path.exists(so_path):
    spec = importlib.util.spec_from_file_location("swiglu_dynamic_quant_sycl", so_path)
    sycl_ext = importlib.util.module_from_spec(spec)
    sys.modules["swiglu_dynamic_quant_sycl"] = sycl_ext
    spec.loader.exec_module(sycl_ext)
else:
    sycl_ext = None

@ProviderRegistry.register_vendor_impl("swiglu_dynamic_quant", "sycl_ext")
class SyclExtSwigluDynamicQuantOp(SwigluDynamicQuantOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["sycl_ext"]

    def vendor_impl(self):
        super().vendor_impl()
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )
        self._run_func = self.swiglu_dynamic_quant_run

    def vendor_parser(self):
        if self.dtype in ["bfloat16", "float16"] and self.dst_dtype in ["int8", "float8", "float8_e4m3", "float8_e4m3fn"]:
            pass
        else:
            raise ValueError(
                f"SyclExtSwigluDynamicQuantOp not support dtype {self.dtype} dst_dtype {self.dst_dtype}"
            )

    def swiglu_dynamic_quant_run(self, tensor_mapping):
        if sycl_ext is None:
            raise RuntimeError("swiglu_dynamic_quant_sycl.so not found. Did you run build.sh?")

        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]
        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        sycl_ext.swiglu_dynamic_quant(
            hidden_states,
            smooth_scale,
            quant_tokens,
            per_token_scale
        )

        return quant_tokens, per_token_scale