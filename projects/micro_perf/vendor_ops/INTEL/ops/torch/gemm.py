import pathlib
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
GemmOp = ProviderRegistry.BASE_IMPL_MAPPING["gemm"]

OP_MAPPING = {}

class INTELGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        if self.dtype == "float32":
            torch.set_float32_matmul_precision("highest")
        elif self.dtype == "tfloat32":
            torch.set_float32_matmul_precision("high")

    def __del__(self):
        torch.set_float32_matmul_precision("highest")
        getattr(super(), "__del__", lambda: None)()


OP_MAPPING["torch"] = INTELGemmOp
