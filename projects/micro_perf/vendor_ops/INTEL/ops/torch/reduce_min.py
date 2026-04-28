from xpu_perf.micro_perf.core.op import ProviderRegistry
ReduceMinOp = ProviderRegistry.BASE_IMPL_MAPPING["reduce_min"]


OP_MAPPING = {}


class INTELReduceMinTorchOp(ReduceMinOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"


# Expose reduce_min torch implementation only via provider name "torch".
OP_MAPPING["torch"] = INTELReduceMinTorchOp
