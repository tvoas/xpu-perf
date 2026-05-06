import pathlib

from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseHeadRMSNormDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["head_rms_norm_dynamic_quant"]


@ProviderRegistry.register_vendor_impl("head_rms_norm_dynamic_quant", "torch")
class HeadRMSNormDynamicQuantTorchOp(BaseHeadRMSNormDynamicQuantOp):
    pass
