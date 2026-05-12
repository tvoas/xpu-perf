import pathlib
import importlib.util
from functools import partial

import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry

AddRmsNormDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["add_rms_norm_dynamic_quant"]


_OP_DIR = pathlib.Path(__file__).resolve().parent
_SYCL_SO = _OP_DIR / "add_rms_norm_dynamic_quant_sycl.so"


try:
    _spec = importlib.util.spec_from_file_location(
        "add_rms_norm_dynamic_quant_sycl", str(_SYCL_SO))
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Failed to create import spec for {_SYCL_SO}")
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl(
        "add_rms_norm_dynamic_quant", "sycl_ext")
    class AddRmsNormDynamicQuantSyclExtOp(AddRmsNormDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def vendor_parser(self):
            if self.dtype not in ("float16", "bfloat16"):
                raise ValueError(
                    f"{type(self).__name__} only supports float16/bfloat16, "
                    f"got dtype={self.dtype}"
                )
            if self.dst_dtype != "int8":
                raise ValueError(
                    f"{type(self).__name__} only supports dst_dtype int8, "
                    f"got dst_dtype={self.dst_dtype}"
                )

        def vendor_impl(self):
            super().vendor_impl()
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            residual = tensor_mapping.get("residual", None)
            norm_weight = tensor_mapping["norm_weight"]
            smooth_scale = tensor_mapping["smooth_scale"]
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]
            after_res = tensor_mapping.get("after_res",
                torch.empty_like(hidden_states))
            after_norm = tensor_mapping.get("after_norm",
                torch.empty_like(hidden_states))

            # Pass empty tensor when no residual
            if residual is None:
                residual = torch.empty(0, device=hidden_states.device,
                                       dtype=hidden_states.dtype)

            _sycl_ext.add_rms_norm_dynamic_quant_forward(
                hidden_states, residual, norm_weight, smooth_scale,
                quant_tokens, per_token_scale, after_res, after_norm,
                self.eps)

            if self.output_mode == "none":
                return quant_tokens, per_token_scale
            if self.output_mode == "res":
                return quant_tokens, per_token_scale, after_res
            return quant_tokens, per_token_scale, after_norm

except Exception as e:
    import warnings
    warnings.warn(
        f"Failed to load SYCL add_rms_norm_dynamic_quant extension: {e}")
