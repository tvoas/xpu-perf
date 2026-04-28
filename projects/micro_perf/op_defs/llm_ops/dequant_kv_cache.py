"""LLM op: dequant_kv_cache (alias of store_kv_cache base implementation)."""
from ._common import *
from .store_kv_cache import StoreKVCacheOp

@ProviderRegistry.register_base_impl("dequant_kv_cache", "ComputeEngine")
class DequantKVCacheOp(StoreKVCacheOp):
    pass
