"""LLM op: quant_group_gemm_reduce_sum (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("quant_group_gemm_reduce_sum", "ComputeEngine")
class QuantGroupGemmReduceSumOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError("QuantGroupGemmReduceSumOp only support llm arg_type")

        self.dtype = self.args_dict["dtype"]
        self.dst_dtype = self.args_dict["dst_dtype"]

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]
        self.trans_w = self.args_dict.get("trans_w", False)

    def vendor_parser(self):
        if not self.dtype in ["int8", "float8"]:
            raise ValueError(
                f"QuantGroupGemmReduceSumOp only support int8 or float8, but got {self.dtype}"
            )
        if not self.dst_dtype in ["bfloat16"]:
            raise ValueError(
                f"QuantGroupGemmReduceSumOp only support bfloat16 dst_dtype, but got {self.dst_dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.sp_size, self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.sp_size, self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "weight": OpTensorInfo(
                shape=[self.sp_size, self.new_hidden_size, self.hidden_size] \
                    if self.trans_w else [self.sp_size, self.hidden_size, self.new_hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "weight_scale": OpTensorInfo(
                shape=[self.sp_size, self.new_hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
        }
        self.output_tensor_info = {
            "output": OpTensorInfo(
                shape=[self.num_tokens, self.new_hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
        }

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ]) + calc_tensor_size(self.output_tensor_info["output"])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 2 * self.sp_size * self.num_tokens * self.new_hidden_size * self.hidden_size

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        per_token_scale = tensor_mapping["per_token_scale"]
        weight = tensor_mapping["weight"]
        weight_scale = tensor_mapping["weight_scale"]

        # quant group gemm
        temp_tensor = torch.empty(
            [self.sp_size, self.num_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=hidden_states.device
        )

        for sp_idx in range(self.sp_size):
            cur_tokens = hidden_states[sp_idx]
            cur_tokens_scale = per_token_scale[sp_idx]
            cur_weight = weight[sp_idx]
            cur_weight_scale = weight_scale[sp_idx]

            temp_tensor[sp_idx] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, 
                cur_weight, cur_weight_scale, 
                dst_torch_dtype=self.dst_torch_dtype, 
                trans_w=self.trans_w,
            )

        # reduce sum
        output = torch.sum(temp_tensor, dim=0, keepdim=False)

        return output
            



