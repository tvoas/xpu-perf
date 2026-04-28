"""LLM op: sage_attention_decode_page (alias of flash_attention base implementation)."""
from ._common import *
from .flash_attention import FlashAttentionOp

@ProviderRegistry.register_base_impl("sage_attention_decode_page", "ComputeEngine")
class SageAttentionDecodePageOp(FlashAttentionOp):
    pass
