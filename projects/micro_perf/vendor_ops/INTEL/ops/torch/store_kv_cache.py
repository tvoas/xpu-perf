import torch
from xpu_perf.micro_perf.core.utils import static_quant

from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseStoreKVCacheOp = ProviderRegistry.BASE_IMPL_MAPPING["store_kv_cache"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    _STORE_CONFIGS = [
        triton.Config({"BLOCK_L": block_l}, num_warps=num_warps, num_stages=num_stages)
        for block_l in (16, 64)
        for num_warps in (2, 4)
        for num_stages in (1, 2)
    ]
    _SINGLE_STORE_CONFIGS = [
        triton.Config({"BLOCK_L": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_L": 4}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_L": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_L": 64}, num_warps=4, num_stages=1),
    ]

    @triton.autotune(configs=_STORE_CONFIGS, key=["B", "H", "D", "Q"])
    @triton.jit
    def _fused_quant_store_kv_kernel(
        packed_ptr,
        k_scale_ptr, v_scale_ptr,
        k_cache_ptr, v_cache_ptr,
        B, H, Q, D,
        cache_len_off,
        k_head_off, v_head_off,
        s_t, s_h, s_d,
        c_b, c_h, c_l, c_d,
        max_val: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        l_block = tl.program_id(2)

        l_offs = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        d_offs = tl.arange(0, BLOCK_D)
        l_mask = l_offs < Q
        d_mask = d_offs < D
        mask = l_mask[:, None] & d_mask[None, :]

        token_row = batch_idx * Q + l_offs
        k_src = token_row[:, None] * s_t + (k_head_off + head_idx) * s_h + d_offs[None, :] * s_d
        v_src = token_row[:, None] * s_t + (v_head_off + head_idx) * s_h + d_offs[None, :] * s_d

        scale_off = head_idx * D + d_offs
        k_scale = tl.load(k_scale_ptr + scale_off, mask=d_mask, other=0.0).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + scale_off, mask=d_mask, other=0.0).to(tl.float32)

        k_value = tl.load(packed_ptr + k_src, mask=mask, other=0.0).to(tl.float32)
        v_value = tl.load(packed_ptr + v_src, mask=mask, other=0.0).to(tl.float32)

        k_quant = tl.minimum(tl.maximum(k_value * k_scale[None, :], -max_val), max_val)
        v_quant = tl.minimum(tl.maximum(v_value * v_scale[None, :], -max_val), max_val)

        k_floor = tl.floor(k_quant)
        v_floor = tl.floor(v_quant)
        k_round = tl.floor(k_quant + 0.5)
        v_round = tl.floor(v_quant + 0.5)
        k_tie = (k_quant - k_floor) == 0.5
        v_tie = (v_quant - v_floor) == 0.5
        k_round_i = k_round.to(tl.int32)
        v_round_i = v_round.to(tl.int32)
        k_round = tl.where(k_tie & ((k_round_i % 2) != 0), k_round - 1.0, k_round)
        v_round = tl.where(v_tie & ((v_round_i % 2) != 0), v_round - 1.0, v_round)

        l_dst = cache_len_off + l_offs
        dst = batch_idx * c_b + head_idx * c_h + l_dst[:, None] * c_l + d_offs[None, :] * c_d
        tl.store(k_cache_ptr + dst, k_round.to(tl.int8), mask=mask)
        tl.store(v_cache_ptr + dst, v_round.to(tl.int8), mask=mask)

    def triton_fused_quant_store_kv(
        packed_qkv, k_scale, v_scale, k_cache, v_cache,
        batch_size, kv_head_num, q_len, head_dim,
        k_head_start, v_head_start, cache_len,
    ):
        BLOCK_D = triton.next_power_of_2(head_dim)
        def grid(meta): return (batch_size, kv_head_num, triton.cdiv(q_len, meta["BLOCK_L"]))
        _fused_quant_store_kv_kernel[grid](
            packed_qkv, k_scale, v_scale, k_cache, v_cache,
            batch_size, kv_head_num, q_len, head_dim,
            cache_len, k_head_start, v_head_start,
            packed_qkv.stride(0), packed_qkv.stride(1), packed_qkv.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            max_val=127.0,
            BLOCK_D=BLOCK_D,
        )

    @triton.autotune(configs=_SINGLE_STORE_CONFIGS, key=["B", "H", "D", "Q", "HEAD_OFF"])
    @triton.jit
    def _single_linear_store_kernel(
        packed_ptr,
        scale_ptr,
        cache_ptr,
        B, H, Q, D,
        cache_len_off,
        head_off,
        s_t, s_h, s_d,
        c_b, c_h, c_l, c_d,
        max_val: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
        DO_QUANT: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        l_block = tl.program_id(2)

        l_offs = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        d_offs = tl.arange(0, BLOCK_D)
        l_mask = l_offs < Q
        d_mask = d_offs < D
        mask = l_mask[:, None] & d_mask[None, :]

        token_row = batch_idx * Q + l_offs
        src = token_row[:, None] * s_t + (head_off + head_idx) * s_h + d_offs[None, :] * s_d
        value = tl.load(packed_ptr + src, mask=mask, other=0.0).to(tl.float32)

        if DO_QUANT:
            scale = tl.load(scale_ptr + head_idx * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            value = tl.minimum(tl.maximum(value * scale[None, :], -max_val), max_val)
            value_floor = tl.floor(value)
            value_round = tl.floor(value + 0.5)
            value_tie = (value - value_floor) == 0.5
            value_round_i = value_round.to(tl.int32)
            value = tl.where(value_tie & ((value_round_i % 2) != 0), value_round - 1.0, value_round).to(tl.int8)

        l_dst = cache_len_off + l_offs
        dst = batch_idx * c_b + head_idx * c_h + l_dst[:, None] * c_l + d_offs[None, :] * c_d
        tl.store(cache_ptr + dst, value, mask=mask)

    def triton_single_linear_store(
        packed_qkv, scale, cache,
        batch_size, kv_head_num, q_len, head_dim,
        head_start, cache_len, do_quant,
    ):
        BLOCK_D = triton.next_power_of_2(head_dim)
        scale_arg = scale if scale is not None else cache
        def grid(meta): return (batch_size, kv_head_num, triton.cdiv(q_len, meta["BLOCK_L"]))
        _single_linear_store_kernel[grid](
            packed_qkv, scale_arg, cache,
            batch_size, kv_head_num, q_len, head_dim,
            cache_len, head_start,
            packed_qkv.stride(0), packed_qkv.stride(1), packed_qkv.stride(2),
            cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
            max_val=127.0,
            BLOCK_D=BLOCK_D,
            DO_QUANT=do_quant,
        )

    @triton.autotune(configs=_SINGLE_STORE_CONFIGS, key=["B", "H", "D", "MAX_Q", "BLOCK_SIZE", "HEAD_OFF"])
    @triton.jit
    def _single_paged_store_kernel(
        packed_ptr,
        scale_ptr,
        cache_ptr,
        block_table_ptr,
        q_lens_ptr, cache_lens_ptr, accum_q_lens_ptr,
        B, H, MAX_Q, D,
        BLOCK_SIZE: tl.constexpr,
        head_off,
        s_t, s_h, s_d,
        bt_b, bt_i,
        c_b, c_h, c_l, c_d,
        max_val: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
        DO_QUANT: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        l_block = tl.program_id(2)

        l_offs = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        d_offs = tl.arange(0, BLOCK_D)
        q_len = tl.load(q_lens_ptr + batch_idx)
        cache_len = tl.load(cache_lens_ptr + batch_idx)
        q_start = tl.load(accum_q_lens_ptr + batch_idx)
        l_mask = l_offs < q_len
        d_mask = d_offs < D
        mask = l_mask[:, None] & d_mask[None, :]

        token_row = q_start + l_offs
        src = token_row[:, None] * s_t + (head_off + head_idx) * s_h + d_offs[None, :] * s_d
        value = tl.load(packed_ptr + src, mask=mask, other=0.0).to(tl.float32)

        if DO_QUANT:
            scale = tl.load(scale_ptr + head_idx * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            value = tl.minimum(tl.maximum(value * scale[None, :], -max_val), max_val)
            value_floor = tl.floor(value)
            value_round = tl.floor(value + 0.5)
            value_tie = (value - value_floor) == 0.5
            value_round_i = value_round.to(tl.int32)
            value = tl.where(value_tie & ((value_round_i % 2) != 0), value_round - 1.0, value_round).to(tl.int8)

        position = cache_len + l_offs
        block_idx = position // BLOCK_SIZE
        block_off = position - block_idx * BLOCK_SIZE
        physical_block = tl.load(block_table_ptr + batch_idx * bt_b + block_idx * bt_i, mask=l_mask, other=-1)
        dst = physical_block[:, None] * c_b + head_idx * c_h + block_off[:, None] * c_l + d_offs[None, :] * c_d
        dst_mask = mask & (physical_block[:, None] >= 0)
        tl.store(cache_ptr + dst, value, mask=dst_mask)

    def triton_single_paged_store(
        packed_qkv, scale, cache,
        block_table, q_lens, cache_lens, accum_q_lens,
        batch_size, kv_head_num, max_q_len, head_dim, block_size,
        head_start, paged_cache_layout, do_quant,
    ):
        BLOCK_D = triton.next_power_of_2(head_dim)
        scale_arg = scale if scale is not None else cache
        if paged_cache_layout == "head_major":
            c_b, c_h, c_l, c_d = cache.stride()
        else:
            c_b = cache.stride(0)
            c_l = cache.stride(1)
            c_h = cache.stride(2)
            c_d = cache.stride(3)

        def grid(meta): return (batch_size, kv_head_num, triton.cdiv(max_q_len, meta["BLOCK_L"]))
        _single_paged_store_kernel[grid](
            packed_qkv, scale_arg, cache,
            block_table, q_lens, cache_lens, accum_q_lens,
            batch_size, kv_head_num, max_q_len, head_dim,
            block_size, head_start,
            packed_qkv.stride(0), packed_qkv.stride(1), packed_qkv.stride(2),
            block_table.stride(0), block_table.stride(1),
            c_b, c_h, c_l, c_d,
            max_val=127.0,
            BLOCK_D=BLOCK_D,
            DO_QUANT=do_quant,
        )


@ProviderRegistry.register_vendor_impl("store_kv_cache", "torch")
class StoreKVCacheOp(BaseStoreKVCacheOp):
    """INTEL vendor override for store_kv_cache."""

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.paged_cache_layout = self.args_dict.get("paged_cache_layout", "head_major")
        if self.paged_cache_layout not in ("head_major", "offset_major"):
            raise ValueError("paged_cache_layout must be either head_major or offset_major")

    def vendor_impl(self):
        super().vendor_impl()
        self._update_token_level_io_bytes()
        self._run_func = self.vectorized_store_run

    def _update_token_level_io_bytes(self):
        src_elem_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        dst_elem_size = torch.tensor([], dtype=self.cache_torch_dtype).element_size()
        cache_streams = int(self.store_mode in ("both", "k")) + int(self.store_mode in ("both", "v"))
        kv_tokens_elems = self.num_tokens * self.kv_head_num * self.head_dim * cache_streams

        self.read_bytes = kv_tokens_elems * src_elem_size
        self.write_bytes = kv_tokens_elems * dst_elem_size
        if self.use_quant:
            self.read_bytes += self.kv_head_num * self.head_dim * cache_streams * 4
        if self.cache_type == "paged":
            num_blocks_per_seq = (
                max(self.cache_lens) + self.num_tokens // self.batch_size + self.block_size - 1
            ) // self.block_size
            self.read_bytes += self.batch_size * num_blocks_per_seq * 4
        self.io_bytes = self.read_bytes + self.write_bytes

    def _can_use_vectorized_linear_store(self):
        uniform_lens = all(q_len == self.q_lens[0] for q_len in self.q_lens) and \
            all(cache_len == self.cache_lens[0] for cache_len in self.cache_lens)
        identity_slots = all(slot == batch_idx for batch_idx, slot in enumerate(self.slot_mapping))
        return uniform_lens and identity_slots

    def _can_use_triton_linear_quant_store(self, k_cache, v_cache, k_scale, v_scale):
        return (
            HAS_TRITON
            and self.cache_type == "linear"
            and self.use_quant
            and self.cache_torch_dtype == torch.int8
            and k_cache is not None
            and v_cache is not None
            and k_scale is not None
            and v_scale is not None
            and self._can_use_vectorized_linear_store()
        )

    def _can_use_triton_single_store(self, cache, scale=None):
        return (
            HAS_TRITON
            and cache is not None
            and (
                (not self.use_quant and self.cache_torch_dtype == torch.bfloat16)
                or (self.use_quant and self.cache_torch_dtype == torch.int8 and scale is not None)
            )
        )

    def vectorized_store_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        k_cache = tensor_mapping.get("k_cache")
        v_cache = tensor_mapping.get("v_cache")
        k_scale = tensor_mapping.get("k_scale")
        v_scale = tensor_mapping.get("v_scale")
        packed_qkv = packed_qkv.view(-1, self.total_head_num, self.head_dim)

        if self.cache_type == "paged":
            return self._paged_store(packed_qkv, k_cache, v_cache, k_scale, v_scale, tensor_mapping)

        if self.cache_type == "linear":
            q_len = self.q_lens[0]
            cache_len = self.cache_lens[0]
            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            if self._can_use_triton_linear_quant_store(k_cache, v_cache, k_scale, v_scale):
                triton_fused_quant_store_kv(
                    packed_qkv, k_scale, v_scale, k_cache, v_cache,
                    self.batch_size, self.kv_head_num, q_len, self.head_dim,
                    k_head_start, v_head_start, cache_len,
                )
                return k_cache, v_cache

            if self._can_use_vectorized_linear_store():
                used_triton = False
                if self._can_use_triton_single_store(k_cache, k_scale):
                    triton_single_linear_store(
                        packed_qkv, k_scale, k_cache,
                        self.batch_size, self.kv_head_num, q_len, self.head_dim,
                        k_head_start, cache_len, self.use_quant,
                    )
                    used_triton = True
                if self._can_use_triton_single_store(v_cache, v_scale):
                    triton_single_linear_store(
                        packed_qkv, v_scale, v_cache,
                        self.batch_size, self.kv_head_num, q_len, self.head_dim,
                        v_head_start, cache_len, self.use_quant,
                    )
                    used_triton = True
                if used_triton:
                    return k_cache, v_cache

            if not self._can_use_vectorized_linear_store() or (self.use_quant and q_len > 32):
                return self._per_batch_store(packed_qkv, k_cache, v_cache, k_scale, v_scale)

            src_all = packed_qkv.view(self.batch_size, q_len, self.total_head_num, self.head_dim)
            src_k = src_all[:, :, k_head_start:k_head_end, :]
            src_v = src_all[:, :, v_head_start:v_head_end, :]
            src_k = src_k.transpose(1, 2)
            src_v = src_v.transpose(1, 2)
            cache_end = cache_len + q_len

            if self.use_quant:
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
            src_k = packed_qkv[q_offset:q_offset + q_len, k_head_start:k_head_end, :]
            src_v = packed_qkv[q_offset:q_offset + q_len, v_head_start:v_head_end, :]

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

    def _paged_store(self, packed_qkv, k_cache, v_cache, k_scale, v_scale, tensor_mapping):
        k_head_start = self.q_head_num
        v_head_start = self.q_head_num + self.kv_head_num
        used_triton = False

        if self._can_use_triton_single_store(k_cache, k_scale):
            triton_single_paged_store(
                packed_qkv, k_scale, k_cache,
                tensor_mapping["block_table"], tensor_mapping["q_lens"],
                tensor_mapping["cache_lens"], tensor_mapping["accum_q_lens"],
                self.batch_size, self.kv_head_num, self.max_q_len, self.head_dim, self.block_size,
                k_head_start, self.paged_cache_layout, self.use_quant,
            )
            used_triton = True
        if self._can_use_triton_single_store(v_cache, v_scale):
            triton_single_paged_store(
                packed_qkv, v_scale, v_cache,
                tensor_mapping["block_table"], tensor_mapping["q_lens"],
                tensor_mapping["cache_lens"], tensor_mapping["accum_q_lens"],
                self.batch_size, self.kv_head_num, self.max_q_len, self.head_dim, self.block_size,
                v_head_start, self.paged_cache_layout, self.use_quant,
            )
            used_triton = True
        if used_triton:
            return k_cache, v_cache

        device = packed_qkv.device
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

        phys = torch.cat(phys_list)
        offsets = torch.cat(off_list)
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
                if self.paged_cache_layout == "head_major":
                    k_cache[phys, :, offsets, :] = k_q
                else:
                    k_cache[phys, offsets, :, :] = k_q
            if v_cache is not None:
                if scale_v is None:
                    raise ValueError("v_scale is required when storing quantized v_cache")
                v_q = src_v.float().mul(scale_v).clamp_(-max_val, max_val).round_().to(self.cache_torch_dtype)
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
        return k_cache, v_cache
