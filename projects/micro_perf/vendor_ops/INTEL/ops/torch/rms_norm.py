from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseRMSNormOp = ProviderRegistry.BASE_IMPL_MAPPING["rms_norm"]


@ProviderRegistry.register_vendor_impl("rms_norm", "torch")
class RMSNormTorchOp(BaseRMSNormOp):
    pass
