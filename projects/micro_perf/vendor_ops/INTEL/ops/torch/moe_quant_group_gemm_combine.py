from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseMoeQuantGroupGemmCombineOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_quant_group_gemm_combine"]


@ProviderRegistry.register_vendor_impl("moe_quant_group_gemm_combine", "torch")
class MoeQuantGroupGemmCombineTorchOp(BaseMoeQuantGroupGemmCombineOp):
    pass
