import pathlib
from xpu_perf.micro_perf.core.op import ProviderRegistry
MoeGatingGemmOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_gating_gemm"]


@ProviderRegistry.register_vendor_impl("moe_gating_gemm", "torch")
class MoeGatingGemmTorchOp(MoeGatingGemmOp):
    """Override vendor_parser to accept bfloat16/float16 input with float32 output."""

    def vendor_parser(self):
        self.dst_dtype = self.args_dict.get("dst_dtype", "float32")

        if self.dtype in ("float16", "bfloat16", "float32") and self.dst_dtype in ("float32", "float16"):
            pass
        else:
            raise ValueError(
                f"MoeGatingGemmTorchOp supports float16/bfloat16/float32-->float32/float16, "
                f"but got {self.dtype}-->{self.dst_dtype}"
            )
