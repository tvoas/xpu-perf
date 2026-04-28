"""LLM op: scale_dynamic_quant (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("scale_dynamic_quant", "ComputeEngine")
class ScaleDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"ScaleDynamicQuantOp only supports llm arg_type, but got {self.arg_type}"
            )

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        # 以下参数决定当前 scale_dynamic_quant 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        self.dst_dtype = self.args_dict.get("dst_dtype", "int8")

    def vendor_parser(self):
        if self.dtype in ("bfloat16", "float16") \
            and self.dst_dtype == "int8":
            pass
        else:
            raise ValueError(
                f"{type(self).__name__} only support bfloat16/float16 -> int8, "
                f"but got dtype={self.dtype}, dst_dtype={self.dst_dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)        

        self.input_tensor_info = {}
        self.output_tensor_info = {}

        self.input_tensor_info["hidden_states"] = OpTensorInfo(
            shape=[self.num_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )
        self.input_tensor_info["smooth_scale"] = OpTensorInfo(
            shape=[self.hidden_size], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )

        self.output_tensor_info["quant_tokens"] = OpTensorInfo(
            shape=[self.num_tokens, self.hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )
        self.output_tensor_info["per_token_scale"] = OpTensorInfo(
            shape=[self.num_tokens], 
            dtype=torch.float32, 
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

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

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
        smooth_scale = tensor_mapping["smooth_scale"]

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            hidden_states, 
            smooth_scale, 
            self.dst_torch_dtype
        )

        return quant_tokens, per_token_scale




