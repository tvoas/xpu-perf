import pathlib
from functools import partial
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
AddRmsNormOp = ProviderRegistry.BASE_IMPL_MAPPING["add_rms_norm"]


try:
    import vllm_xpu_kernels._C

    @ProviderRegistry.register_vendor_impl("add_rms_norm", "vllm_xpu_kernels")
    class VLLMXPUKernelsAddRmsNormOp(AddRmsNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=False,
            )

        def add_rms_norm_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            residual = tensor_mapping["residual"]
            norm_weight = tensor_mapping["norm_weight"]

            # fused_add_rms_norm: input += residual, then rms_norm(input) in-place
            # It modifies input (adds residual) and returns normalized result in input
            torch.ops._C.fused_add_rms_norm(hidden_states, residual, norm_weight, self.eps)

            # After the call, hidden_states contains the residual sum,
            # and the normalized output is written back to hidden_states
            return hidden_states

except Exception:
    pass
