from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseMoeSoftmaxTopkOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_softmax_topk"]


@ProviderRegistry.register_vendor_impl("moe_softmax_topk", "torch")
class MoeSoftmaxTopkTorchOp(BaseMoeSoftmaxTopkOp):
    pass
