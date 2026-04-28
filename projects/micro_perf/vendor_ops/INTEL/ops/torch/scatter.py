from xpu_perf.micro_perf.core.op import ProviderRegistry
ScatterOp = ProviderRegistry.BASE_IMPL_MAPPING["scatter"]


OP_MAPPING = {}


class INTELScatterTorchOp(ScatterOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"


OP_MAPPING["torch"] = INTELScatterTorchOp
