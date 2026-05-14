from functools import partial
import torch
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, get_attn_info
from xpu_perf.micro_perf.core.op import ProviderRegistry
BaseDequantKVCacheOp = ProviderRegistry.BASE_IMPL_MAPPING["dequant_kv_cache"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ── Triton fused dequant kernel (K + V in one launch) ───────────────────────
if HAS_TRITON:
    _DEQUANT_CONFIGS = [
        triton.Config({"BLOCK_L": block_l}, num_warps=w, num_stages=s)
        for block_l in (1,)
        for w in (1, 2, 4)
        for s in (1, 2)
    ]

    @triton.autotune(configs=_DEQUANT_CONFIGS, key=["B", "H", "L", "D"])
    @triton.jit
    def _fused_dequant_kv_kernel(
        k_src_ptr, v_src_ptr, k_scale_ptr, v_scale_ptr, k_dst_ptr, v_dst_ptr,
        B, H, L,
        D: tl.constexpr,
        s_src_b, s_src_h, s_src_l, s_src_d,
        s_dst_b, s_dst_h, s_dst_l, s_dst_d,
        BLOCK_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_l_blocks = tl.cdiv(L, BLOCK_L)
        l_block = pid % num_l_blocks
        tmp = pid // num_l_blocks
        h_idx = tmp % H
        b_idx = tmp // H

        l_offs = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        d_offs = tl.arange(0, BLOCK_D)
        l_mask = l_offs < L
        d_mask = d_offs < D
        mask = l_mask[:, None] & d_mask[None, :]

        k_scale = tl.load(k_scale_ptr + h_idx * D + d_offs, mask=d_mask, other=0.0).to(tl.bfloat16)
        v_scale = tl.load(v_scale_ptr + h_idx * D + d_offs, mask=d_mask, other=0.0).to(tl.bfloat16)

        src_off = (
            b_idx * s_src_b
            + h_idx * s_src_h
            + l_offs[:, None] * s_src_l
            + d_offs[None, :] * s_src_d
        )
        dst_off = (
            b_idx * s_dst_b
            + h_idx * s_dst_h
            + l_offs[:, None] * s_dst_l
            + d_offs[None, :] * s_dst_d
        )

        k_value = tl.load(k_src_ptr + src_off, mask=mask, other=0.0).to(tl.bfloat16)
        v_value = tl.load(v_src_ptr + src_off, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(k_dst_ptr + dst_off, k_value * k_scale[None, :], mask=mask)
        tl.store(v_dst_ptr + dst_off, v_value * v_scale[None, :], mask=mask)

    def triton_fused_dequant_kv(k_src, v_src, k_scale, v_scale, k_dst, v_dst, kv_len):
        """k_src/v_src: [B,H,max_seq,D], scales: [H,D] bf16, dsts: bf16."""
        B, H = k_src.shape[0], k_src.shape[1]
        D = k_src.shape[3]
        BLOCK_D = triton.next_power_of_2(D)
        def grid(meta): return (B * H * triton.cdiv(kv_len, meta["BLOCK_L"]),)
        _fused_dequant_kv_kernel[grid](
            k_src, v_src, k_scale, v_scale, k_dst, v_dst,
            B, H, kv_len, D,
            k_src.stride(0), k_src.stride(1), k_src.stride(2), k_src.stride(3),
            k_dst.stride(0), k_dst.stride(1), k_dst.stride(2), k_dst.stride(3),
            BLOCK_D=BLOCK_D,
        )


@ProviderRegistry.register_vendor_impl("dequant_kv_cache", "torch")
class DequantKVCacheOp(BaseDequantKVCacheOp):
    """Standalone INTEL dequant_kv_cache implementation.

    Upstream removed this op class; we keep it here for int8/float8
    KV-cache dequantization benchmarking on Intel GPUs.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def _can_use_linear_batch_vectorization(self):
        uniform_kv_lens = all(b_kv_len == self.kv_lens[0] for b_kv_len in self.kv_lens)
        identity_slots = all(slot == batch_idx for batch_idx, slot in enumerate(self.slot_mapping))
        return uniform_kv_lens and identity_slots

    def _can_use_triton_linear_dequant(self):
        return HAS_TRITON and self.dtype in ["int8", "float8"] and self._can_use_linear_batch_vectorization()

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm", "batch_llm"]:
            raise ValueError

        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if self.attn_mode not in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)

        # src (quant) dtype
        self.dtype = self.args_dict.get("dtype", "int8")
        if self.dtype not in ["int8", "float8"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dequant target dtype
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")
        if self.dst_dtype not in ["bfloat16"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.quant_mode = self.args_dict.get("quant_mode", "static")
        if self.quant_mode not in ["static"]:
            raise ValueError

        # all tokens with same head/head_dim element pos share one scale
        if self.quant_mode == "static":
            self.quant_scale_shape = [self.kv_head_num, self.head_dim]

        self.input_tensor_info = {
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.kv_lens, dtype=dtype, device=device
                ),
            ),
            "k_scale": OpTensorInfo(
                shape=self.quant_scale_shape,
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
            "v_scale": OpTensorInfo(
                shape=self.quant_scale_shape,
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
        }
        self.output_tensor_info = {}

        if self.cache_type == "linear":
            self.input_tensor_info["slot_mapping"] = OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.slot_mapping, dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        elif self.cache_type == "paged":
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.target_batch_size, self.target_per_seq_num_block],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.block_table, dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["kv_lens"])
            + calc_tensor_size(self.input_tensor_info["k_scale"])
            + calc_tensor_size(self.input_tensor_info["v_scale"])
        )

        if self.cache_type == "linear":
            self.read_bytes += (
                calc_tensor_size(self.input_tensor_info["slot_mapping"])
                + calc_tensor_size(self.input_tensor_info["k_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
                + calc_tensor_size(self.input_tensor_info["v_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
                + calc_tensor_size(self.output_tensor_info["dequant_v_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
            )

        elif self.cache_type == "paged":
            self.read_bytes += (
                calc_tensor_size(self.input_tensor_info["block_table"])
                / self.target_batch_size
                / self.target_per_seq_num_block
                * self.num_kv_blocks
                + calc_tensor_size(self.input_tensor_info["k_cache"])
                / self.total_cache_blocks
                / self.block_size
                * self.num_kv_tokens
                + calc_tensor_size(self.input_tensor_info["v_cache"])
                / self.total_cache_blocks
                / self.block_size
                * self.num_kv_tokens
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"])
                / self.total_cache_blocks
                / self.block_size
                * self.num_kv_tokens
                + calc_tensor_size(self.output_tensor_info["dequant_v_cache"])
                / self.total_cache_blocks
                / self.block_size
                * self.num_kv_tokens
            )

        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )

        self._run_func = self.dequant_kv_cache_run

    def dequant_kv_cache_run(self, tensor_mapping):
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        dequant_k_cache = tensor_mapping["dequant_k_cache"]
        dequant_v_cache = tensor_mapping["dequant_v_cache"]

        k_scale = tensor_mapping["k_scale"]
        v_scale = tensor_mapping["v_scale"]

        k_s = k_scale.to(self.dst_torch_dtype).view(1, self.kv_head_num, 1, self.head_dim)
        v_s = v_scale.to(self.dst_torch_dtype).view(1, self.kv_head_num, 1, self.head_dim)

        if self.cache_type == "paged":
            device = k_cache.device
            phys_list = []
            off_list = []
            for batch_idx in range(self.batch_size):
                kv_len = self.kv_lens[batch_idx]
                positions = torch.arange(kv_len, device=device)
                block_indices = (positions // self.block_size).long()
                block_offsets = positions % self.block_size
                bt = torch.tensor(self.block_table[batch_idx], dtype=torch.long, device=device)
                phys_list.append(bt[block_indices])
                off_list.append(block_offsets)

            phys = torch.cat(phys_list)
            offsets = torch.cat(off_list)

            k_s_2d = k_s.view(self.kv_head_num, self.head_dim)
            v_s_2d = v_s.view(self.kv_head_num, self.head_dim)

            dequant_k_cache[phys, :, offsets, :] = k_cache[phys, :, offsets, :].to(self.dst_torch_dtype) * k_s_2d
            dequant_v_cache[phys, :, offsets, :] = v_cache[phys, :, offsets, :].to(self.dst_torch_dtype) * v_s_2d

            return dequant_k_cache, dequant_v_cache

        if self.cache_type == "linear":
            kv_len = self.kv_lens[0]
            uniform_kv_lens = all(b_kv_len == kv_len for b_kv_len in self.kv_lens)

            if self._can_use_triton_linear_dequant():
                k_scale_bf16 = k_scale.to(self.dst_torch_dtype)
                v_scale_bf16 = v_scale.to(self.dst_torch_dtype)
                triton_fused_dequant_kv(
                    k_cache, v_cache,
                    k_scale_bf16, v_scale_bf16,
                    dequant_k_cache, dequant_v_cache,
                    kv_len,
                )
            elif uniform_kv_lens and self._can_use_linear_batch_vectorization():
                if self.batch_size <= 4:
                    # Vectorized: good for small batch
                    src_k = k_cache[:, :, :kv_len, :].to(self.dst_torch_dtype)
                    src_v = v_cache[:, :, :kv_len, :].to(self.dst_torch_dtype)
                    dequant_k_cache[:, :, :kv_len, :] = src_k * k_s
                    dequant_v_cache[:, :, :kv_len, :] = src_v * v_s
                else:
                    # Per-batch: avoids huge temporaries for large batch
                    for b in range(self.batch_size):
                        slot_idx = self.slot_mapping[b]
                        src_k = k_cache[slot_idx:slot_idx+1, :, :kv_len, :].to(self.dst_torch_dtype)
                        src_v = v_cache[slot_idx:slot_idx+1, :, :kv_len, :].to(self.dst_torch_dtype)
                        dequant_k_cache[slot_idx:slot_idx+1, :, :kv_len, :] = src_k * k_s
                        dequant_v_cache[slot_idx:slot_idx+1, :, :kv_len, :] = src_v * v_s
            else:
                for b in range(self.batch_size):
                    slot_idx = self.slot_mapping[b]
                    b_kv_len = self.kv_lens[b]
                    src_k = k_cache[slot_idx:slot_idx+1, :, :b_kv_len, :].to(self.dst_torch_dtype)
                    src_v = v_cache[slot_idx:slot_idx+1, :, :b_kv_len, :].to(self.dst_torch_dtype)
                    dequant_k_cache[slot_idx:slot_idx+1, :, :b_kv_len, :] = src_k * k_s
                    dequant_v_cache[slot_idx:slot_idx+1, :, :b_kv_len, :] = src_v * v_s

        return dequant_k_cache, dequant_v_cache
