from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseMoeGatherOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_gather"]


@ProviderRegistry.register_vendor_impl("moe_gather", "torch")
class MoeGatherTorchOp(BaseMoeGatherOp):
    pass
