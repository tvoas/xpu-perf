import pathlib
import torch
import random
from functools import partial
from itertools import combinations

from xpu_perf.micro_perf.core.op import ProviderRegistry, BasicOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size

try:
    torch.ops.torch_ipex.mm_w8a8

    @ProviderRegistry.register_vendor_impl("moe_quant_matmul", "ipex")
    class MoeQuantMatmulOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            # src_dtype
            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["int8"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            # dst_dtype
            self.dst_dtype = self.args_dict["dst_dtype"]
            if not self.dst_dtype in ["float16"]:
                raise NotImplementedError
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

            # pre-defined attrs
            self.sp_size = self.args_dict.get("sp_size", 1)
            self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
            self.hidden_size = self.args_dict["hidden_size"]
            self.new_hidden_size = self.args_dict["new_hidden_size"]

            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                "per_token_scale": OpTensorInfo(
                    shape=[self.num_tokens], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ), 
                "expert_weight": OpTensorInfo(
                    shape=[self.new_hidden_size, self.hidden_size], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                "expert_scale": OpTensorInfo(
                    shape=[self.new_hidden_size], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ),
                "zero_points": OpTensorInfo(
                    shape=[1],
                    dtype=torch.int8, 
                    device=self.backend.get_torch_device_name()),
            }
            self.output_tensor_info = {
                "y": OpTensorInfo(
                    shape=[self.num_tokens, self.new_hidden_size], 
                    dtype=self.dst_torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

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

            self.algo_size = 0
            self.bus_size = 0

            self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size


            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )

            # run func
            self._run_func = self.moe_quant_matmul_run


        def moe_quant_matmul_run(self, tensor_mapping):
            # get pre-allocated input tensors
            hidden_states = tensor_mapping["hidden_states"]
            per_token_scale = tensor_mapping["per_token_scale"]
            expert_weight = tensor_mapping["expert_weight"]
            expert_scale = tensor_mapping["expert_scale"]
            zero_points = tensor_mapping["zero_points"]
            y = tensor_mapping["y"]

            expert_weight = expert_weight.transpose(0, 1)
            torch.ops.torch_ipex.mm_w8a8(
                hidden_states,
                per_token_scale,
                None,
                expert_weight,
                expert_scale,
                zero_points,
                y
            )

            return y


except Exception:
    pass
