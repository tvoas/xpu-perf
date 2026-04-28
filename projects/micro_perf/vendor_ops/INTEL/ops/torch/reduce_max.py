from xpu_perf.micro_perf.core.op import ProviderRegistry
ReduceMaxOp = ProviderRegistry.BASE_IMPL_MAPPING["reduce_max"]


OP_MAPPING = {}


class INTELReduceMaxTorchOp(ReduceMaxOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"


OP_MAPPING["torch"] = INTELReduceMaxTorchOp
