import torch
from xpu_perf.micro_perf.core.utils import static_quant

from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseStoreKVCacheOp = ProviderRegistry.BASE_IMPL_MAPPING["store_kv_cache"]


@ProviderRegistry.register_vendor_impl("store_kv_cache", "torch")
class StoreKVCacheOp(BaseStoreKVCacheOp):
    """INTEL vendor override for store_kv_cache.

    Optimizes the base implementation by replacing the per-batch Python
    for-loop with vectorized tensor operations across all batches.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl(self):
        super().vendor_impl()
        self._run_func = self.vectorized_store_run

    def _can_use_vectorized_linear_store(self):
        uniform_lens = all(q_len == self.q_lens[0] for q_len in self.q_lens) and \
            all(cache_len == self.cache_lens[0] for cache_len in self.cache_lens)
        identity_slots = all(slot == batch_idx for batch_idx, slot in enumerate(self.slot_mapping))
        return uniform_lens and identity_slots

    def vectorized_store_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        k_cache = tensor_mapping.get("k_cache")
        v_cache = tensor_mapping.get("v_cache")

        k_scale = tensor_mapping.get("k_scale")
        v_scale = tensor_mapping.get("v_scale")

        # Reshape from [num_tokens, total_dim] to [num_tokens, total_heads, head_dim]
        packed_qkv = packed_qkv.view(-1, self.total_head_num, self.head_dim)

        if self.cache_type == "paged":
            return self._paged_store(packed_qkv, k_cache, v_cache, k_scale, v_scale)

        if self.cache_type == "linear":
            q_len = self.q_lens[0]
            cache_len = self.cache_lens[0]

            # Fall back for non-uniform batches, non-identity slot mapping, or
            # quant + large prefill where vectorization creates large fp32 temporaries.
            if not self._can_use_vectorized_linear_store() or (self.use_quant and q_len > 32):
                return self._per_batch_store(packed_qkv, k_cache, v_cache, k_scale, v_scale)

            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            # packed_qkv: [num_tokens, total_head_num, head_dim]
            # Reshape to [batch_size, q_len, head_num, head_dim]
            src_all = packed_qkv.view(self.batch_size, q_len, self.total_head_num, self.head_dim)

            # Extract K and V: [batch_size, q_len, kv_head_num, head_dim]
            src_k = src_all[:, :, k_head_start:k_head_end, :]
            src_v = src_all[:, :, v_head_start:v_head_end, :]

            # Transpose to cache layout: [batch_size, kv_head_num, q_len, head_dim]
            # No .contiguous() needed — copy_() handles non-contiguous sources
            src_k = src_k.transpose(1, 2)
            src_v = src_v.transpose(1, 2)

            cache_end = cache_len + q_len

            if self.use_quant:
                # Only reached for small q_len (decode), safe to materialize
                scale_k = k_scale.view(1, self.kv_head_num, 1, self.head_dim) if k_scale is not None else None
                scale_v = v_scale.view(1, self.kv_head_num, 1, self.head_dim) if v_scale is not None else None

                if self.cache_torch_dtype == torch.int8:
                    max_val = 127.0
                else:
                    raise ValueError(f"Unsupported cache dtype: {self.cache_torch_dtype}")

                if k_cache is not None:
                    if scale_k is None:
                        raise ValueError("k_scale is required when storing quantized k_cache")
                    k_quant = src_k.float().mul_(scale_k).clamp_(-max_val, max_val)
                    if self.cache_torch_dtype == torch.int8:
                        k_quant.round_()
                    k_cache[:, :, cache_len:cache_end, :].copy_(k_quant.to(self.cache_torch_dtype))

                if v_cache is not None:
                    if scale_v is None:
                        raise ValueError("v_scale is required when storing quantized v_cache")
                    v_quant = src_v.float().mul_(scale_v).clamp_(-max_val, max_val)
                    if self.cache_torch_dtype == torch.int8:
                        v_quant.round_()
                    v_cache[:, :, cache_len:cache_end, :].copy_(v_quant.to(self.cache_torch_dtype))
            else:
                if k_cache is not None:
                    k_cache[:, :, cache_len:cache_end, :].copy_(src_k)
                if v_cache is not None:
                    v_cache[:, :, cache_len:cache_end, :].copy_(src_v)

        return k_cache, v_cache

    def _per_batch_store(self, packed_qkv, k_cache, v_cache, k_scale, v_scale):
        """Per-batch linear cache store."""
        k_head_start = self.q_head_num
        k_head_end = self.q_head_num + self.kv_head_num
        v_head_start = self.q_head_num + self.kv_head_num
        v_head_end = self.q_head_num + self.kv_head_num * 2

        for batch_idx in range(self.batch_size):
            slot_idx = self.slot_mapping[batch_idx]
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]
            cache_end = cache_len + q_len

            # [q_len, kv_head_num, head_dim]
            src_k = packed_qkv[q_offset:q_offset + q_len, k_head_start:k_head_end, :]
            src_v = packed_qkv[q_offset:q_offset + q_len, v_head_start:v_head_end, :]

            # Quantize and transpose to [kv_head_num, q_len, head_dim]
            if self.use_quant:
                if k_cache is not None:
                    if k_scale is None:
                        raise ValueError("k_scale is required when storing quantized k_cache")
                    k_cache[slot_idx, :, cache_len:cache_end, :].copy_(
                        static_quant(src_k, k_scale, self.cache_torch_dtype).transpose(0, 1))
                if v_cache is not None:
                    if v_scale is None:
                        raise ValueError("v_scale is required when storing quantized v_cache")
                    v_cache[slot_idx, :, cache_len:cache_end, :].copy_(
                        static_quant(src_v, v_scale, self.cache_torch_dtype).transpose(0, 1))
            else:
                if k_cache is not None:
                    k_cache[slot_idx, :, cache_len:cache_end, :].copy_(
                        src_k.transpose(0, 1).to(self.cache_torch_dtype))
                if v_cache is not None:
                    v_cache[slot_idx, :, cache_len:cache_end, :].copy_(
                        src_v.transpose(0, 1).to(self.cache_torch_dtype))

        return k_cache, v_cache

    def _paged_store(self, packed_qkv, k_cache, v_cache, k_scale, v_scale):
        """Vectorized paged cache store: single scatter instead of per-token loop."""
        device = packed_qkv.device
        k_head_start = self.q_head_num
        v_head_start = self.q_head_num + self.kv_head_num

        # Build scatter indices: map each token → (physical_block, block_offset)
        phys_list = []
        off_list = []
        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]
            positions = torch.arange(q_len, device=device) + cache_len
            block_indices = (positions // self.block_size).long()
            block_offsets = positions % self.block_size
            bt = torch.tensor(self.block_table[batch_idx], dtype=torch.long, device=device)
            phys_list.append(bt[block_indices])
            off_list.append(block_offsets)

        phys = torch.cat(phys_list)     # [num_tokens]
        offsets = torch.cat(off_list)   # [num_tokens]

        # Extract K, V: [num_tokens, kv_head_num, head_dim]
        src_k = packed_qkv[:, k_head_start:k_head_start + self.kv_head_num, :]
        src_v = packed_qkv[:, v_head_start:v_head_start + self.kv_head_num, :]

        if self.use_quant:
            scale_k = k_scale.view(1, self.kv_head_num, self.head_dim) if k_scale is not None else None
            scale_v = v_scale.view(1, self.kv_head_num, self.head_dim) if v_scale is not None else None
            if self.cache_torch_dtype == torch.int8:
                max_val = 127.0
            else:
                raise ValueError(f"Unsupported cache dtype: {self.cache_torch_dtype}")

            if k_cache is not None:
                if scale_k is None:
                    raise ValueError("k_scale is required when storing quantized k_cache")
                k_q = src_k.float().mul(scale_k).clamp_(-max_val, max_val).round_().to(self.cache_torch_dtype)
                k_cache[phys, :, offsets, :] = k_q
            if v_cache is not None:
                if scale_v is None:
                    raise ValueError("v_scale is required when storing quantized v_cache")
                v_q = src_v.float().mul(scale_v).clamp_(-max_val, max_val).round_().to(self.cache_torch_dtype)
                v_cache[phys, :, offsets, :] = v_q
        else:
            if k_cache is not None:
                k_cache[phys, :, offsets, :] = src_k.to(self.cache_torch_dtype)
            if v_cache is not None:
                v_cache[phys, :, offsets, :] = src_v.to(self.cache_torch_dtype)

        return k_cache, v_cache
