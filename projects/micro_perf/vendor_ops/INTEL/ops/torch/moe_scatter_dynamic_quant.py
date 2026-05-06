from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseMoeScatterDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_scatter_dynamic_quant"]


@ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "torch")
class MoeScatterDynamicQuantTorchOp(BaseMoeScatterDynamicQuantOp):
    pass
