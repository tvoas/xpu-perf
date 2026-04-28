import pathlib

from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseHeadRMSNormDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["head_rms_norm_dynamic_quant"]


class HeadRMSNormDynamicQuantTorchOp(BaseHeadRMSNormDynamicQuantOp):
    pass


OP_MAPPING = {"torch": HeadRMSNormDynamicQuantTorchOp}
