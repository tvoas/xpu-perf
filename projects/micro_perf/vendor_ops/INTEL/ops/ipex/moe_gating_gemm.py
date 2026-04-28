import pathlib
import torch
import random
from functools import partial
from itertools import combinations

from xpu_perf.micro_perf.core.op import ProviderRegistry, BasicOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size

try:
    torch.ops.torch_ipex.specialized_gating_gemm

    @ProviderRegistry.register_vendor_impl("moe_gating_gemm", "ipex")
    class MoeGatingGemmIpexOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            # src_dtype
            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["float16", "bfloat16", "float32"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            # dst_dtype
            self.dst_dtype = self.args_dict.get("dst_dtype", "float32")
            if not self.dst_dtype in ["float32", "float16"]:
                raise NotImplementedError
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

            # pre-defined attrs
            self.num_experts = self.args_dict["num_experts"]
            self.sp_size = self.args_dict.get("sp_size", 1)
            self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
            self.hidden_size = self.args_dict["hidden_size"]
        
            # input/output tensors
            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                "gating_weight": OpTensorInfo(
                    shape=[self.num_experts, self.hidden_size], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ),
                "gating_weight_trans": OpTensorInfo(
                    shape=[self.hidden_size, self.num_experts], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                )
            }
            self.output_tensor_info = {
                "gating_output": OpTensorInfo(
                    shape=[self.num_tokens, self.num_experts], 
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

            self.read_bytes = calc_tensor_size(self.input_tensor_info["hidden_states"]) + calc_tensor_size(self.input_tensor_info["gating_weight"]) 
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.num_experts

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )

            # run func
            self._run_func = self.moe_gating_gemm_run


        def moe_gating_gemm_run(self, tensor_mapping):
            ntok = tensor_mapping["hidden_states"].shape[0]
            gating_output = tensor_mapping["gating_output"]
            if ntok == 40 or ntok == 80 or ntok == 4096:
                gating_output = torch.ops.torch_ipex.specialized_gating_gemm(
                    tensor_mapping["gating_weight"],
                    tensor_mapping["hidden_states"], 
                    tensor_mapping["gating_output"]
                )
            else:
                gating_output = torch.matmul(
                    tensor_mapping["hidden_states"],
                    tensor_mapping["gating_weight_trans"]
                    ).type(self.dst_torch_dtype)
            return gating_output


except Exception:
    pass
