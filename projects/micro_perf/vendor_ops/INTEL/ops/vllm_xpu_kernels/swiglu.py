import pathlib
from functools import partial
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
SwigluOp = ProviderRegistry.BASE_IMPL_MAPPING["swiglu"]


try:
    import vllm_xpu_kernels._C

    @ProviderRegistry.register_vendor_impl("swiglu", "vllm_xpu_kernels")
    class VLLMXPUKernelsSwigluOp(SwigluOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

        def vendor_impl_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            output_tokens = tensor_mapping["output_tokens"]

            # silu_and_mul: out[i] = silu(input[i, :H]) * input[i, H:]
            torch.ops._C.silu_and_mul(output_tokens, hidden_states)
            return output_tokens

except Exception:
    pass
