"""LLM op: flash_attention (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("flash_attention", "ComputeEngine")
class FlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)
        
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.softmax_scale = self.head_dim ** (-0.5)
        self.is_causal = True

        # 以下参数决定当前 flash_attention 的计算类型
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")
        self.cache_dtype = self.args_dict.get("cache_dtype", "bfloat16")
        self.qk_compute_dtype = self.args_dict.get("qk_compute_dtype", self.dtype)
        self.pv_compute_dtype = self.args_dict.get("pv_compute_dtype", self.dtype)

        self.flops_calc()


    def flops_calc(self):
        self.calc_flops = 0
        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]
            kv_len = self.kv_lens[batch_idx]

            """
            q_len = 8, cache_len = 0, kv_len = 8
            total = kv_len * kv_len = 64
            valid = (cache_len + 1 + kv_len) * q_len / 2 = 36
            ratio = 36 / 64 = 0.5625
            * - - - - - - - 
            * * - - - - - - 
            * * * - - - - - 
            * * * * - - - - 
            * * * * * - - - 
            * * * * * * - - 
            * * * * * * * - 
            * * * * * * * *

            q_len = 4, cache_len = 4, kv_len = 8
            total = kv_len * kv_len = 64
            valid = (cache_len + 1 + kv_len) * q_len / 2 = 26
            ratio = 26 / 64 = 0.40625
            - - - - - - - - 
            - - - - - - - - 
            - - - - - - - - 
            - - - - - - - - 
            * * * * * - - - 
            * * * * * * - - 
            * * * * * * * - 
            * * * * * * * *
            """

            valid_parts = kv_len * kv_len
            if self.is_causal:
                valid_parts = (cache_len + 1 + kv_len) * q_len / 2
            else:
                valid_parts = q_len * kv_len

            # p = q * v, bf16/int8/fp8 batch_gemm
            # o = p * v, bf16/int8/fp8 batch_gemm
            self.calc_flops += 2 * (self.q_head_num * self.head_dim * valid_parts * 2)





    def vendor_parser(self):
        if self.dst_dtype != "bfloat16":
            raise ValueError("FlashAttentionOp dst_dtype must be bfloat16.")

        """
        1. 基础 bf16 attention 实现
        2. 基础 bf16 attention 实现 + int8 cache, 
           如果是prefill, 一般可以 attn 前反量化必要的 kv_cache tokens
           如果是decode, 一般都是在 attn 内部实时反量化
           厂商也可以在 attention 内实时反量化
        """
        if self.dtype == "bfloat16" \
            and self.cache_dtype == "bfloat16" \
            and self.qk_compute_dtype == "bfloat16" \
            and self.pv_compute_dtype == "bfloat16":
            self.use_quant = False
        elif self.dtype == "bfloat16" \
            and self.cache_dtype == "int8" \
            and self.qk_compute_dtype == "bfloat16" \
            and self.pv_compute_dtype == "bfloat16":
            self.use_quant = True
        else:
            raise ValueError(
                f"current impl not supported, please check "
                "dtype/cache_dtype/compute_dtype.")


    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)
        self.cache_torch_dtype = get_torch_dtype(self.cache_dtype)
        self.qk_compute_torch_dtype = get_torch_dtype(self.qk_compute_dtype)
        self.pv_compute_torch_dtype = get_torch_dtype(self.pv_compute_dtype)

        self.input_tensor_info = {}
        self.output_tensor_info = {}


        """
        输入的q, packed模式, 需要考虑可能是
        """
        self.input_tensor_info["q"] = OpTensorInfo(
            shape=[self.num_tokens, self.q_head_num, self.head_dim], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name()
        )
        
        """
        确定当前的 num_tokens 中的具体的组成信息
        """
        self.attn_info_tensors = {
            "q_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ), 
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            ), 
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.kv_lens, dtype=dtype, device=device)
            ), 
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ), 
            "accum_cache_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_cache_lens, dtype=dtype, device=device)
            ), 
            "accum_kv_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_kv_lens, dtype=dtype, device=device)
            ), 
        }
        self.input_tensor_info.update(self.attn_info_tensors)



        """
        kv cache信息, 根据 block_size 决定是 linear 还是 paged
        """
        if self.cache_type == "linear":
            self.input_tensor_info["slot_mapping"] = OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: \
                    torch.tensor(self.slot_mapping, dtype=dtype, device=device)
            )
            cache_shape = [self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim]
        elif self.cache_type == "paged":
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.target_batch_size, self.target_per_seq_num_block],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: \
                    torch.tensor(self.block_table, dtype=dtype, device=device)
            )
            cache_shape = [self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim]
        self.input_tensor_info["k_cache"] = OpTensorInfo(
            shape=cache_shape, 
            dtype=self.cache_torch_dtype, 
            device=self.backend.get_torch_device_name(), 
        )
        self.input_tensor_info["v_cache"] = OpTensorInfo(
            shape=cache_shape, 
            dtype=self.cache_torch_dtype, 
            device=self.backend.get_torch_device_name(), 
        )
        
        """
        根据 kv cache 是否需要量化确定量化参数
        """
        if self.use_quant:
            quant_scale_shape = [self.kv_head_num, self.head_dim]
            self.input_tensor_info["k_scale"] = OpTensorInfo(
                shape=quant_scale_shape, 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            )
            self.input_tensor_info["v_scale"] = OpTensorInfo(
                shape=quant_scale_shape, 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            )
        
    
        self.output_tensor_info["out"] = OpTensorInfo(
            shape=[self.num_tokens, self.q_head_num, self.head_dim], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name()
        )

        # calculator
        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        cache_bytes = calc_tensor_size(self.input_tensor_info["k_cache"]) \
                    + calc_tensor_size(self.input_tensor_info["v_cache"])

        # 提前反量化需要额外的内存空间
        if self.use_quant:            
            extra_cache_bytes = cache_bytes // get_torch_dtype_size(self.cache_torch_dtype) \
                                * get_torch_dtype_size(self.qk_compute_torch_dtype)
            self.tensor_size += extra_cache_bytes

        self.read_bytes = self.input_tensor_size - cache_bytes
        if self.cache_type == "linear":
            self.read_bytes += \
                cache_bytes / self.batch_size / self.max_kv_len * self.num_kv_tokens
        elif self.cache_type == "paged":
            self.read_bytes += \
                cache_bytes / self.total_cache_blocks * self.num_kv_blocks
        
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        
        # specify create input/output tensors func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        raise NotImplementedError
        


"""
******************************************
gemm & group_gemm & moe_ops
******************************************
"""

