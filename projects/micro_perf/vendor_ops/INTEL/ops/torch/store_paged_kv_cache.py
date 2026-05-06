import pathlib
from functools import partial
import torch
import random
import importlib.util
from xpu_perf.micro_perf.core.op import ProviderRegistry, BasicOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size

_utils_path = pathlib.Path(__file__).resolve().parent.parent / "utils.py"
_spec = importlib.util.spec_from_file_location("intel_ops_utils", _utils_path)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
generate_prefill_data = _utils.generate_prefill_data
generate_prefill_session_cache_data = _utils.generate_prefill_session_cache_data
generate_decode_data = _utils.generate_decode_data


@ProviderRegistry.register_vendor_impl("store_paged_kv_cache", "torch")
class StorePagedKVCacheOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict.get("cache_dtype", self.args_dict.get("dst_dtype", "int8"))
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.torch_dst_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.block_size = self.args_dict["block_size"]
        self.total_block_num = self.args_dict["total_block_num"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

        self.mode = self.args_dict.get("attn_mode", self.args_dict.get("mode", "prefill"))
        if self.mode == "prefill":
            # [q_seq_len, total_head_num, head_dim]
            self.batch_size = 1
            self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_prefill_data(
                    self.q_seq_len,
                    self.cache_len
                )

        elif self.mode == "prefill_session_cache":
            # [accumed_num_tokens, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_prefill_session_cache_data(
                    self.batch_size,
                    self.q_seq_len,
                    self.cache_len
                )

        elif self.mode == "decode":
            # [batch_size * q_seq_len, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_decode_data(
                    self.batch_size,
                    self.q_seq_len,
                    self.cache_len
                )

        else:
            raise NotImplementedError

        # accum q_lens
        self.num_tokens = sum(self.q_lens)
        # accum cache_lens
        self.num_cache_tokens = sum(self.cache_lens)
        # max q_len + cache_len
        self.max_kv_len = max(self.kv_lens)
        self.max_q_len = max(self.q_lens)
        self.max_block_num_per_seq = (self.max_kv_len + self.block_size - 1) // self.block_size

        if self.max_block_num_per_seq * self.batch_size > self.total_block_num:
            raise ValueError

        block_tables_lst = random.sample(range(self.total_block_num), self.max_block_num_per_seq * self.batch_size)

        self.input_tensor_info = {
            "packed_qkv": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "q_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ),
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ),
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            ),
            "block_table": OpTensorInfo(
                shape=[self.batch_size, self.max_block_num_per_seq],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(block_tables_lst, dtype=dtype, device=device).reshape([self.batch_size, self.max_block_num_per_seq])
            ),
            "k_cache": OpTensorInfo(
                shape=[self.total_block_num, self.block_size, self.kv_head_num, self.head_dim],
                dtype=self.torch_dst_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "v_cache": OpTensorInfo(
                shape=[self.total_block_num, self.block_size, self.kv_head_num, self.head_dim],
                dtype=self.torch_dst_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            ),
            "k_scale": OpTensorInfo(
                shape=[self.total_block_num, self.block_size, self.kv_head_num],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
            "v_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }

        self.output_tensor_info = {

        }

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * 2 * self.kv_head_num + \
            calc_tensor_size(self.input_tensor_info["q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["accum_q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["cache_lens"]) + \
            calc_tensor_size(self.input_tensor_info["block_table"]) / self.total_block_num * self.num_tokens + \
            calc_tensor_size(self.input_tensor_info["k_scale"]) / self.total_block_num  / self.block_size * self.num_tokens + \
            calc_tensor_size(self.input_tensor_info["v_scale"])

        self.write_bytes = \
            calc_tensor_size(self.input_tensor_info["k_cache"]) / self.total_block_num  / self.block_size * self.num_tokens + \
            calc_tensor_size(self.input_tensor_info["v_cache"]) / self.total_block_num / self.block_size * self.num_tokens

        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

        # run func
        self._run_func = self.store_kv_cache_run

    def store_kv_cache_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        block_table = tensor_mapping["block_table"]
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]

            k_data = packed_qkv[q_offset:q_offset + q_len, self.q_head_num:self.q_head_num + self.kv_head_num, :]
            v_data = packed_qkv[q_offset:q_offset + q_len, self.q_head_num + self.kv_head_num:, :]

            for token_idx in range(q_len):
                global_pos = cache_len + token_idx
                block_idx = global_pos // self.block_size
                block_offset = global_pos % self.block_size
                physical_block = block_table[batch_idx, block_idx]

                k_tok = k_data[token_idx]
                v_tok = v_data[token_idx]
                k_abs = k_tok.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
                k_s = k_abs / 127.0
                k_cache[physical_block, block_offset] = (k_tok.float() / k_s).round().clamp(-128, 127).to(k_cache.dtype)
                v_cache[physical_block, block_offset] = (v_tok.float() / v_tok.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) * 127.0).round().clamp(-128, 127).to(v_cache.dtype)

        return k_cache, v_cache
