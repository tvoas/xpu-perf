"""LLM op: moe_quant_matmul (alias of quant_matmul base implementation)."""
from ._common import *
from .quant_matmul import QuantMatmulOp

@ProviderRegistry.register_base_impl("moe_quant_matmul", "ComputeEngine")
class MoeQuantMatmulOp(QuantMatmulOp):
    pass
