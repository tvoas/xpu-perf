"""LLM op: sage_attention_v1 (alias of flash_attention base implementation)."""
from ._common import *
from .flash_attention import FlashAttentionOp

@ProviderRegistry.register_base_impl("sage_attention_v1", "ComputeEngine")
class SageAttentionV1Op(FlashAttentionOp):
    pass
