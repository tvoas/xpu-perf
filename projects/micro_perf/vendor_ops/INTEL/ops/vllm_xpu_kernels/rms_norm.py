import pathlib
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
RMSNormOp = ProviderRegistry.BASE_IMPL_MAPPING["rms_norm"]


try:
    import torch
    import vllm_xpu_kernels._C

    @ProviderRegistry.register_vendor_impl("rms_norm", "vllm_xpu_kernels")
    class VLLMXPUKernelsRMSNormOp(RMSNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["vllm_xpu_kernels"]

            if self.add_residual:
                self._create_tensors_func = partial(
                    self._create_in_out_tensors,
                    create_inputs=True,
                    create_outputs=False,
                )
            else:
                self._create_tensors_func = partial(
                    self._create_in_out_tensors,
                    create_inputs=True,
                    create_outputs=True,
                )

        def rms_norm_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            orig_shape = src.shape
            src = src.view(-1, src.shape[-1])

            if self.add_residual:
                residual = tensor_mapping["residual"]
                residual = residual.view(-1, residual.shape[-1])
                torch.ops._C.fused_add_rms_norm(src, residual, weight, self.epsilon)
                return src.view(orig_shape)
            else:
                dst = tensor_mapping["dst"]
                torch.ops._C.rms_norm(dst, src, weight, self.epsilon)
                return dst.view(orig_shape)

except Exception:
    pass
