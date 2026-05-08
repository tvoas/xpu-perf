"""
SYCL extension provider for store_kv_cache.

Uses custom fused SYCL kernels:
  - int8: bf16 -> fp32 -> scale -> clamp -> round -> int8 in ONE kernel
  - bf16: bf16 -> bf16 permute+copy in ONE kernel

Same linear cache layout as torch path, so we reuse base tensor setup
and only override the run function.
"""
import torch
import importlib.util
import os
from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.utils import static_quant
StoreKVCacheOp = ProviderRegistry.BASE_IMPL_MAPPING["store_kv_cache"]


# Load the compiled SYCL extension
_SYCL_SO = os.path.join(os.path.dirname(__file__), "store_kv_cache_sycl.so")

try:
    _spec = importlib.util.spec_from_file_location("store_kv_cache_sycl", _SYCL_SO)
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("store_kv_cache", "sycl_ext")
    class SYCLExtStoreKVCacheOp(StoreKVCacheOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def vendor_parser(self):
            """Extend base parser to accept float8/float8_e4m3 cache_dtype."""
            if self.dtype == "bfloat16" and self.cache_dtype in ("float8", "float8_e4m3"):
                self.use_quant = True
            else:
                super().vendor_parser()

        def vendor_impl(self):
            """Reuse base class tensor setup, override IO model and run func."""
            super().vendor_impl()

            # IO model: count only the cache streams requested by store_mode.
            src_elem_size = torch.tensor([], dtype=self.torch_dtype).element_size()
            dst_elem_size = torch.tensor([], dtype=self.cache_torch_dtype).element_size()
            cache_streams = int(self.store_mode in ("both", "k")) + int(self.store_mode in ("both", "v"))
            kv_tokens_elems = self.num_tokens * self.kv_head_num * self.head_dim * cache_streams

            self.read_bytes = kv_tokens_elems * src_elem_size
            self.write_bytes = kv_tokens_elems * dst_elem_size
            if self.use_quant:
                # per-channel static scale: kv_head_num * head_dim for each requested cache stream
                self.read_bytes += self.kv_head_num * self.head_dim * cache_streams * 4  # float32 scales
            if self.cache_type == "paged":
                # one int32 block_table lookup per token in fallback path
                self.read_bytes += self.num_tokens * 4
            self.io_bytes = self.read_bytes + self.write_bytes

            self._run_func = self.sycl_store_kv_cache_run

        def sycl_store_kv_cache_run(self, tensor_mapping):
            packed_qkv = tensor_mapping["packed_qkv"]
            k_cache = tensor_mapping.get("k_cache")
            v_cache = tensor_mapping.get("v_cache")

            q_len = self.q_lens[0]
            cache_len = self.cache_lens[0]
            bs = self.batch_size

            k_head_start = self.q_head_num
            uniform_lens = all(bq_len == q_len for bq_len in self.q_lens) and \
                all(b_cache_len == cache_len for b_cache_len in self.cache_lens)
            identity_slots = self.cache_type != "linear" or \
                all(slot == batch_idx for batch_idx, slot in enumerate(self.slot_mapping))

            # SYCL kernel stores both K and V in linear cache only;
            # fall back to torch logic for partial modes or paged cache.
            if k_cache is None or v_cache is None or self.cache_type == "paged" or not uniform_lens or not identity_slots:
                packed_qkv_3d = packed_qkv.view(-1, self.total_head_num, self.head_dim)
                v_head_start = k_head_start + self.kv_head_num

                for batch_idx in range(bs):
                    slot_idx = self.slot_mapping[batch_idx] if self.cache_type == "linear" else batch_idx
                    bq_len = self.q_lens[batch_idx]
                    q_offset = self.accum_q_lens[batch_idx]
                    b_cache_len = self.cache_lens[batch_idx]

                    for token_idx in range(bq_len):
                        src_token = packed_qkv_3d[q_offset + token_idx]

                        if self.cache_type == "paged":
                            global_pos = b_cache_len + token_idx
                            block_idx = global_pos // self.block_size
                            block_offset = global_pos % self.block_size
                            physical_block = self.block_table[batch_idx][block_idx]
                            if k_cache is not None:
                                src_k = src_token[k_head_start:k_head_start + self.kv_head_num, :]
                                if self.use_quant:
                                    src_k = static_quant(src_k.reshape(1, -1), tensor_mapping["k_scale"], self.cache_torch_dtype).view(self.kv_head_num, self.head_dim)
                                else:
                                    src_k = src_k.to(self.cache_torch_dtype)
                                k_cache[physical_block, :, block_offset, :] = src_k
                            if v_cache is not None:
                                src_v = src_token[v_head_start:v_head_start + self.kv_head_num, :]
                                if self.use_quant:
                                    src_v = static_quant(src_v.reshape(1, -1), tensor_mapping["v_scale"], self.cache_torch_dtype).view(self.kv_head_num, self.head_dim)
                                else:
                                    src_v = src_v.to(self.cache_torch_dtype)
                                v_cache[physical_block, :, block_offset, :] = src_v
                        else:
                            pos = b_cache_len + token_idx
                            if k_cache is not None:
                                src_k = src_token[k_head_start:k_head_start + self.kv_head_num, :]
                                if self.use_quant:
                                    src_k = static_quant(src_k.reshape(1, -1), tensor_mapping["k_scale"], self.cache_torch_dtype).view(self.kv_head_num, self.head_dim)
                                else:
                                    src_k = src_k.to(self.cache_torch_dtype)
                                k_cache[slot_idx, :, pos, :] = src_k
                            if v_cache is not None:
                                src_v = src_token[v_head_start:v_head_start + self.kv_head_num, :]
                                if self.use_quant:
                                    src_v = static_quant(src_v.reshape(1, -1), tensor_mapping["v_scale"], self.cache_torch_dtype).view(self.kv_head_num, self.head_dim)
                                else:
                                    src_v = src_v.to(self.cache_torch_dtype)
                                v_cache[slot_idx, :, pos, :] = src_v

                return k_cache, v_cache

            # Reshape packed_qkv from [num_tokens, total_dim] to [num_tokens, total_heads, head_dim]
            packed_qkv = packed_qkv.view(-1, self.total_head_num, self.head_dim)

            if self.use_quant:
                k_scale = tensor_mapping["k_scale"]
                v_scale = tensor_mapping["v_scale"]
                if self.cache_dtype in ("float8", "float8_e4m3"):
                    _sycl_ext.store_kv_cache_fp8(
                        packed_qkv, k_cache, v_cache,
                        k_scale, v_scale,
                        k_head_start, self.kv_head_num,
                        bs, q_len, cache_len
                    )
                else:
                    _sycl_ext.store_kv_cache_int8(
                        packed_qkv, k_cache, v_cache,
                        k_scale, v_scale,
                        k_head_start, self.kv_head_num,
                        bs, q_len, cache_len
                    )
            else:
                _sycl_ext.store_kv_cache_bf16(
                    packed_qkv, k_cache, v_cache,
                    k_head_start, self.kv_head_num,
                    bs, q_len, cache_len
                )

            return k_cache, v_cache

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL store_kv_cache extension: {e}")
