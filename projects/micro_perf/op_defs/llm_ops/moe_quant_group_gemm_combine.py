"""LLM op: moe_quant_group_gemm_combine (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_quant_group_gemm_combine", "ComputeEngine")
class MoeQuantGroupGemmCombineOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"MoeQuantGroupGemmCombineOp only support llm arg_type, but got {self.arg_type}"
            )

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]

        # moe info
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        # parallel info
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)

        # resiual info
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.sp_rank = self.args_dict.get("sp_rank", 0)
        self.res_scale = self.args_dict.get("res_scale", 1.0)

        if self.sp_size > 1:
            self.num_res_tokens_per_rank = (self.num_tokens + self.sp_size - 1) // self.sp_size
            self.res_token_start = self.sp_rank * self.num_res_tokens_per_rank
            self.res_token_end = min(self.res_token_start + self.num_res_tokens_per_rank, self.num_tokens)
        else:
            self.num_res_tokens_per_rank = 1
            self.res_token_start = 0
            self.res_token_end = self.num_tokens

        # get moe token disptch info
        self.num_scatter_tokens, \
        self.num_scatter_tokens_per_rank, \
        self.num_experts_per_rank, \
        self.experts_start_idx, \
        self.experts_end_idx, \
        self.all_select_experts, \
        self.all_select_weights, \
        self.dispatch_tokens, \
        self.used_src_tokens, \
        self.expert_dispatch_tokens, \
        self.expert_dispatch_weights, \
        self.scatter_token_id, \
        self.scatter_token_weight, \
        self.expert_dispatch_token_count, \
        self.expert_dispatch_token_offset = get_moe_tokens_info(
            self.num_tokens, self.num_experts, self.topk, 
            ep_size=self.ep_size, ep_rank=self.ep_rank
        )


        # 以下参数决定当前的 moe_quant_group_gemm_combine 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "int8")
        self.w_dtype = self.args_dict.get("w_dtype", "int8")
        self.compute_dtype = self.args_dict.get("compute_dtype", "int8")
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")



    def vendor_parser(self):
        if self.dtype == "int8" \
            and self.w_dtype == "int8" \
            and self.compute_dtype == "int8" \
            and self.dst_dtype == "bfloat16":
            pass
        else:
            raise ValueError(
                f"MoeQuantGroupGemmCombineOp base impl not support: "
                f"dtype={self.dtype}, w_dtype={self.w_dtype}, "
                f"compute_dtype={self.compute_dtype}, dst_dtype={self.dst_dtype}"
            )



    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.w_torch_dtype = get_torch_dtype(self.w_dtype)
        self.compute_torch_dtype = get_torch_dtype(self.compute_dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {}
        self.output_tensor_info = {}

        # hidden_states
        self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
            shape=[self.dispatch_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )
        self.input_tensor_info["per_token_scale"] = OpTensorInfo(
            shape=[self.dispatch_tokens], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )

        # experts_weight
        self.input_tensor_info["experts_weight"] = OpTensorInfo(
            shape=[self.num_experts_per_rank, self.new_hidden_size, self.hidden_size], 
            dtype=self.w_torch_dtype, 
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros
        )
        self.input_tensor_info["experts_scale"] = OpTensorInfo(
            shape=[self.num_experts_per_rank, self.new_hidden_size], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )

        # expert distribution info
        self.input_tensor_info["experts_token_count"] = OpTensorInfo(
            shape=[self.num_experts_per_rank], 
            dtype=torch.int32, 
            device=self.backend.get_torch_device_name(),
            creator=partial(create_from_list, data=self.expert_dispatch_token_count)
        )
        self.input_tensor_info["experts_token_offset"] = OpTensorInfo(
            shape=[self.num_experts_per_rank], 
            dtype=torch.int32, 
            device=self.backend.get_torch_device_name(),
            creator=partial(create_from_list, data=self.expert_dispatch_token_offset)
        )
        self.input_tensor_info["scatter_token_id"] = OpTensorInfo(
            shape=[self.dispatch_tokens], 
            dtype=torch.int32, 
            device=self.backend.get_torch_device_name(),
            creator=partial(create_from_list, data=self.scatter_token_id)
        )
        self.input_tensor_info["scatter_token_weight"] = OpTensorInfo(
            shape=[self.dispatch_tokens], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=partial(create_from_list, data=self.scatter_token_weight)
        )

        # res input
        self.input_tensor_info["residual_tokens"] = OpTensorInfo(
            shape=[self.num_res_tokens_per_rank, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )

        temp_scatter_tokens = OpTensorInfo(
            shape=[self.dispatch_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )

        self.output_tensor_info["convergent_tokens"] = OpTensorInfo(
            shape=[self.num_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros
        )

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        scatter_tokens_bytes = calc_tensor_size(temp_scatter_tokens)
        self.tensor_size += scatter_tokens_bytes

        self.read_bytes = self.input_tensor_size
        self.write_bytes = scatter_tokens_bytes
        self.io_bytes = self.read_bytes + self.write_bytes


        self.calc_flops = 2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        experts_weight = tensor_mapping["experts_weight"]
        experts_scale = tensor_mapping["experts_scale"]

        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]
        
        residual_tokens = tensor_mapping["residual_tokens"]

        # get pre-allocated output tensors
        convergent_tokens = tensor_mapping["convergent_tokens"]
        convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens * self.res_scale

        
        new_scatter_tokens = torch.empty(
            size=[self.dispatch_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )
        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = experts_token_offset[expert_idx]
            cur_token_end = cur_token_start + experts_token_count[expert_idx]

            cur_tokens = scatter_tokens[cur_token_start:cur_token_end]
            cur_tokens_scale = per_token_scale[cur_token_start:cur_token_end]

            cur_weight = experts_weight[expert_idx]
            cur_weight_scale = experts_scale[expert_idx]

            new_scatter_tokens[cur_token_start:cur_token_end] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, 
                cur_weight, cur_weight_scale, 
                dst_torch_dtype=self.dst_torch_dtype,
                trans_w=True
            )

        convergent_tokens.index_add_(
            0, scatter_token_id, 
            (new_scatter_tokens * scatter_token_weight.unsqueeze(-1)).to(self.dst_torch_dtype)
        )

        return convergent_tokens



