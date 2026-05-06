from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseMoeQuantGroupGemmOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_quant_group_gemm"]


@ProviderRegistry.register_vendor_impl("moe_quant_group_gemm", "torch")
class MoeQuantGroupGemmTorchOp(BaseMoeQuantGroupGemmOp):
    pass
