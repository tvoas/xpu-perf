from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseQuantMatmulOp = ProviderRegistry.BASE_IMPL_MAPPING["quant_matmul"]


@ProviderRegistry.register_vendor_impl("quant_matmul", "torch")
class QuantMatmulTorchOp(BaseQuantMatmulOp):
    pass
