"""LLM op: store_paged_kv_cache (alias of store_kv_cache base implementation)."""
from ._common import *
from .store_kv_cache import StoreKVCacheOp

@ProviderRegistry.register_base_impl("store_paged_kv_cache", "ComputeEngine")
class StorePagedKVCacheOp(StoreKVCacheOp):
    pass
