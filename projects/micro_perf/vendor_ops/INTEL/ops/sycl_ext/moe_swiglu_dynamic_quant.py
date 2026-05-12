import os
import sys
import importlib.util
import torch
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
MoeSwigluDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_swiglu_dynamic_quant"]

# Dynamically load the localized SYCL extension
so_path = os.path.join(os.path.dirname(__file__), "moe_swiglu_dynamic_quant_sycl.so")
if os.path.exists(so_path):
    spec = importlib.util.spec_from_file_location("moe_swiglu_dynamic_quant_sycl", so_path)
    sycl_ext = importlib.util.module_from_spec(spec)
    sys.modules["moe_swiglu_dynamic_quant_sycl"] = sycl_ext
    spec.loader.exec_module(sycl_ext)
else:
    sycl_ext = None

@ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "sycl_ext")
class SyclExtMoeSwigluDynamicQuantOp(MoeSwigluDynamicQuantOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["sycl_ext"]


    def vendor_impl(self):
        super().vendor_impl()
        import torch
        from xpu_perf.micro_perf.core.utils import OpTensorInfo
        
        self.scatter_expert_ids = [
            expert_idx
            for expert_idx, token_count in enumerate(self.expert_dispatch_token_count)
            for _ in range(token_count)
        ]
        
        self.input_tensor_info.update({
            "experts_token_start": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset[:self.num_experts_per_rank], dtype=dtype, device=device)
            ),
            "scatter_expert_ids": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_expert_ids, dtype=dtype, device=device)
            )
        })

        from xpu_perf.micro_perf.core.utils import calc_tensor_size
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.total_experts_num_val = len(self.expert_dispatch_token_count)
        self.max_token_num_val = max(self.expert_dispatch_token_count) if self.total_experts_num_val > 0 else 0

        self._run_func = self.moe_swiglu_dynamic_quant_run

    def vendor_parser(self):
        if self.dtype in ["bfloat16", "float16"] and self.dst_dtype in ["int8", "float8", "float8_e4m3", "float8_e4m3fn"]:
            pass
        else:
            raise ValueError(
                f"SyclExtMoeSwigluDynamicQuantOp not support dtype {self.dtype} dst_dtype {self.dst_dtype}"
            )

    def moe_swiglu_dynamic_quant_run(self, tensor_mapping):
        if sycl_ext is None:
            raise RuntimeError("moe_swiglu_dynamic_quant_sycl.so not found. Did you run build.sh?")

        scatter_tokens = tensor_mapping["scatter_tokens"]
        smooth_scale = tensor_mapping["experts_smooth_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_start = tensor_mapping["experts_token_start"]
        scatter_expert_ids = tensor_mapping["scatter_expert_ids"]

        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        sycl_ext.moe_swiglu_dynamic_quant(
            scatter_tokens,
            smooth_scale,
            experts_token_count,
            experts_token_start,
            scatter_expert_ids,
            quant_tokens,
            per_token_scale,
            self.total_experts_num_val,
            self.max_token_num_val
        )

        return quant_tokens, per_token_scale
