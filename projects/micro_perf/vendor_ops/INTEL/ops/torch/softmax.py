from xpu_perf.micro_perf.core.op import ProviderRegistry
SoftmaxOp = ProviderRegistry.BASE_IMPL_MAPPING["softmax"]


@ProviderRegistry.register_vendor_impl("softmax", "torch")
class INTELSoftmaxTorchOp(SoftmaxOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"
