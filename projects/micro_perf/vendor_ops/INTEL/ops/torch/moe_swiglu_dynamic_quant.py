from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseMoeSwigluDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_swiglu_dynamic_quant"]


@ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "torch")
class MoeSwigluDynamicQuantTorchOp(BaseMoeSwigluDynamicQuantOp):
    pass
