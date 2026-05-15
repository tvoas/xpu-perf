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
            self.paged_cache_layout = args_dict.get("paged_cache_layout", "head_major")

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
                # SYCL paged kernel reads bs * num_blocks_per_seq entries from block_table
                num_blocks_per_seq = (max(self.cache_lens) + self.num_tokens // self.batch_size + self.block_size - 1) // self.block_size
                self.read_bytes += self.batch_size * num_blocks_per_seq * 4  # int32 block_table
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

            # SYCL fast path supports uniform lens and identity linear slots.
            if not uniform_lens or not identity_slots:
                packed_qkv_3d = packed_qkv.view(-1, self.total_head_num, self.head_dim)
                v_head_start = k_head_start + self.kv_head_num

                if self.cache_type == "paged":
                    # Vectorized paged scatter: build index arrays, single write
                    device = packed_qkv.device
                    phys_list = []
                    off_list = []
                    for batch_idx in range(bs):
                        bq_len = self.q_lens[batch_idx]
                        b_cache_len = self.cache_lens[batch_idx]
                        positions = torch.arange(bq_len, device=device) + b_cache_len
                        block_indices = (positions // self.block_size).long()
                        block_offsets = positions % self.block_size
                        bt = torch.tensor(self.block_table[batch_idx], dtype=torch.long, device=device)
                        phys_list.append(bt[block_indices])
                        off_list.append(block_offsets)

                    phys = torch.cat(phys_list)
                    offsets = torch.cat(off_list)

                    src_k = packed_qkv_3d[:, k_head_start:k_head_start + self.kv_head_num, :]
                    src_v = packed_qkv_3d[:, v_head_start:v_head_start + self.kv_head_num, :]

                    if self.use_quant:
                        if k_cache is not None:
                            k_q = static_quant(src_k, tensor_mapping["k_scale"], self.cache_torch_dtype)
                            if self.paged_cache_layout == "head_major":
                                k_cache[phys, :, offsets, :] = k_q
                            else:
                                k_cache[phys, offsets, :, :] = k_q
                        if v_cache is not None:
                            v_q = static_quant(src_v, tensor_mapping["v_scale"], self.cache_torch_dtype)
                            if self.paged_cache_layout == "head_major":
                                v_cache[phys, :, offsets, :] = v_q
                            else:
                                v_cache[phys, offsets, :, :] = v_q
                    else:
                        if k_cache is not None:
                            if self.paged_cache_layout == "head_major":
                                k_cache[phys, :, offsets, :] = src_k.to(self.cache_torch_dtype)
                            else:
                                k_cache[phys, offsets, :, :] = src_k.to(self.cache_torch_dtype)
                        if v_cache is not None:
                            if self.paged_cache_layout == "head_major":
                                v_cache[phys, :, offsets, :] = src_v.to(self.cache_torch_dtype)
                            else:
                                v_cache[phys, offsets, :, :] = src_v.to(self.cache_torch_dtype)
                else:
                    # Linear non-uniform/non-identity: per-batch slice copy
                    for batch_idx in range(bs):
                        slot_idx = self.slot_mapping[batch_idx]
                        bq_len = self.q_lens[batch_idx]
                        q_offset = self.accum_q_lens[batch_idx]
                        b_cache_len = self.cache_lens[batch_idx]

                        src_k = packed_qkv_3d[q_offset:q_offset + bq_len, k_head_start:k_head_start + self.kv_head_num, :]
                        src_v = packed_qkv_3d[q_offset:q_offset + bq_len, v_head_start:v_head_start + self.kv_head_num, :]

                        if self.use_quant:
                            if k_cache is not None:
                                k_cache[slot_idx, :, b_cache_len:b_cache_len + bq_len, :] = static_quant(
                                    src_k, tensor_mapping["k_scale"], self.cache_torch_dtype).transpose(0, 1)
                            if v_cache is not None:
                                v_cache[slot_idx, :, b_cache_len:b_cache_len + bq_len, :] = static_quant(
                                    src_v, tensor_mapping["v_scale"], self.cache_torch_dtype).transpose(0, 1)
                        else:
                            if k_cache is not None:
                                k_cache[slot_idx, :, b_cache_len:b_cache_len + bq_len, :] = src_k.transpose(0, 1).to(self.cache_torch_dtype)
                            if v_cache is not None:
                                v_cache[slot_idx, :, b_cache_len:b_cache_len + bq_len, :] = src_v.transpose(0, 1).to(self.cache_torch_dtype)

                return k_cache, v_cache

            # Reshape packed_qkv from [num_tokens, total_dim] to [num_tokens, total_heads, head_dim]
            packed_qkv = packed_qkv.view(-1, self.total_head_num, self.head_dim)

            if self.cache_type == "paged":
                block_table = tensor_mapping["block_table"]
                offset_major = self.paged_cache_layout == "offset_major"
                if self.use_quant:
                    if self.cache_dtype in ("float8", "float8_e4m3"):
                        if self.store_mode == "both":
                            _sycl_ext.store_kv_cache_fp8_paged(
                                packed_qkv, k_cache, v_cache,
                                tensor_mapping["k_scale"], tensor_mapping["v_scale"],
                                block_table,
                                k_head_start, self.kv_head_num,
                                bs, q_len, cache_len, offset_major)
                        elif self.store_mode == "k":
                            _sycl_ext.store_kv_cache_fp8_single_paged(
                                packed_qkv, k_cache, tensor_mapping["k_scale"],
                                block_table,
                                k_head_start, self.kv_head_num,
                                bs, q_len, cache_len, offset_major)
                        else:
                            _sycl_ext.store_kv_cache_fp8_single_paged(
                                packed_qkv, v_cache, tensor_mapping["v_scale"],
                                block_table,
                                k_head_start + self.kv_head_num, self.kv_head_num,
                                bs, q_len, cache_len, offset_major)
                    else:
                        if self.store_mode == "both":
                            _sycl_ext.store_kv_cache_int8_paged(
                                packed_qkv, k_cache, v_cache,
                                tensor_mapping["k_scale"], tensor_mapping["v_scale"],
                                block_table,
                                k_head_start, self.kv_head_num,
                                bs, q_len, cache_len, offset_major)
                        elif self.store_mode == "k":
                            _sycl_ext.store_kv_cache_int8_single_paged(
                                packed_qkv, k_cache, tensor_mapping["k_scale"],
                                block_table,
                                k_head_start, self.kv_head_num,
                                bs, q_len, cache_len, offset_major)
                        else:
                            _sycl_ext.store_kv_cache_int8_single_paged(
                                packed_qkv, v_cache, tensor_mapping["v_scale"],
                                block_table,
                                k_head_start + self.kv_head_num, self.kv_head_num,
                                bs, q_len, cache_len, offset_major)
                else:
                    if self.store_mode == "both":
                        _sycl_ext.store_kv_cache_bf16_paged(
                            packed_qkv, k_cache, v_cache,
                            block_table,
                            k_head_start, self.kv_head_num,
                            bs, q_len, cache_len, offset_major)
                    elif self.store_mode == "k":
                        _sycl_ext.store_kv_cache_bf16_single_paged(
                            packed_qkv, k_cache,
                            block_table,
                            k_head_start, self.kv_head_num,
                            bs, q_len, cache_len, offset_major)
                    else:
                        _sycl_ext.store_kv_cache_bf16_single_paged(
                            packed_qkv, v_cache,
                            block_table,
                            k_head_start + self.kv_head_num, self.kv_head_num,
                            bs, q_len, cache_len, offset_major)
                return k_cache, v_cache

            # Linear path: identity slots, uniform lens
            if self.use_quant:
                if self.cache_dtype in ("float8", "float8_e4m3"):
                    if self.store_mode == "both":
                        _sycl_ext.store_kv_cache_fp8(
                            packed_qkv, k_cache, v_cache,
                            tensor_mapping["k_scale"], tensor_mapping["v_scale"],
                            k_head_start, self.kv_head_num,
                            bs, q_len, cache_len
                        )
                    elif self.store_mode == "k":
                        _sycl_ext.store_kv_cache_fp8_single(
                            packed_qkv, k_cache, tensor_mapping["k_scale"],
                            k_head_start, self.kv_head_num,
                            bs, q_len, cache_len
                        )
                    else:
                        _sycl_ext.store_kv_cache_fp8_single(
                            packed_qkv, v_cache, tensor_mapping["v_scale"],
                            k_head_start + self.kv_head_num, self.kv_head_num,
                            bs, q_len, cache_len
                        )
                else:
                    if self.store_mode == "both":
                        _sycl_ext.store_kv_cache_int8(
                            packed_qkv, k_cache, v_cache,
                            tensor_mapping["k_scale"], tensor_mapping["v_scale"],
                            k_head_start, self.kv_head_num,
                            bs, q_len, cache_len
                        )
                    elif self.store_mode == "k":
                        _sycl_ext.store_kv_cache_int8_single(
                            packed_qkv, k_cache, tensor_mapping["k_scale"],
                            k_head_start, self.kv_head_num,
                            bs, q_len, cache_len
                        )
                    else:
                        _sycl_ext.store_kv_cache_int8_single(
                            packed_qkv, v_cache, tensor_mapping["v_scale"],
                            k_head_start + self.kv_head_num, self.kv_head_num,
                            bs, q_len, cache_len
                        )
            else:
                if self.store_mode == "both":
                    _sycl_ext.store_kv_cache_bf16(
                        packed_qkv, k_cache, v_cache,
                        k_head_start, self.kv_head_num,
                        bs, q_len, cache_len
                    )
                elif self.store_mode == "k":
                    _sycl_ext.store_kv_cache_bf16_single(
                        packed_qkv, k_cache,
                        k_head_start, self.kv_head_num,
                        bs, q_len, cache_len
                    )
                else:
                    _sycl_ext.store_kv_cache_bf16_single(
                        packed_qkv, v_cache,
                        k_head_start + self.kv_head_num, self.kv_head_num,
                        bs, q_len, cache_len
                    )

            return k_cache, v_cache

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL store_kv_cache extension: {e}")
