"""LLM op: quant_matmul (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("quant_matmul", "ComputeEngine")
class QuantMatmulOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(f"QuantMatmulOp only support llm arg_type, but got {self.arg_type}")

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]
        self.has_bias = self.args_dict.get("has_bias", False)
        self.transpose_o = self.args_dict.get("transpose_o", False)

        # 以下参数决定当前 quant_matmul 的具体数据类型
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
                f"QuantMatmulOp base impl not support: "
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

        # per_token scale
        self.input_tensor_info["hidden_states"] = OpTensorInfo(
            shape=[self.num_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )
        self.input_tensor_info["per_token_scale"] = OpTensorInfo(
            shape=[self.num_tokens], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )

        # per_channel scale
        # 默认TN, 厂商可以在自己的impl自行选择
        self.input_tensor_info["expert_weight"] = OpTensorInfo(
            shape=[self.new_hidden_size, self.hidden_size], 
            dtype=self.w_torch_dtype, 
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros
        )
        self.input_tensor_info["expert_scale"] = OpTensorInfo(
            shape=[self.new_hidden_size], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )
        if self.has_bias:
            self.input_tensor_info["expert_bias"] = OpTensorInfo(
                shape=[self.new_hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            )

        # output
        self.output_tensor_info["y"] =  OpTensorInfo(
            shape=[self.num_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype
        )


        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        per_token_scale = tensor_mapping["per_token_scale"]
        expert_weight = tensor_mapping["expert_weight"]
        expert_scale = tensor_mapping["expert_scale"]
        if self.has_bias:
            expert_bias = tensor_mapping["expert_bias"]
        else:
            expert_bias = None

        # [num_tokens // sp_size, new_hidden_size]
        # [num_tokens // sp_size, sp_size, new_hidden_size // sp_size]
        # [sp_size, new_hidden_size // sp_size, new_hidden_size // sp_size]
        y = fake_quant_gemm(
            hidden_states, per_token_scale, 
            expert_weight, expert_scale, 
            dst_torch_dtype=self.dst_torch_dtype, 
            trans_w=True
        )
        if self.has_bias:
            y = y + expert_bias.unsqueeze(0)
        if self.transpose_o and self.sp_size > 1:
            y = y.view(self.num_tokens, self.sp_size, self.new_hidden_size // self.sp_size)
            y = y.transpose(0, 1).contiguous().view(self.num_tokens, self.new_hidden_size)
        return y



