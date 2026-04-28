"""LLM op: sage_attention_page (alias of flash_attention base implementation)."""
from ._common import *
from .flash_attention import FlashAttentionOp

@ProviderRegistry.register_base_impl("sage_attention_page", "ComputeEngine")
class SageAttentionPageOp(FlashAttentionOp):
    pass
