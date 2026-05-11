import os
import sys
import importlib.util
import torch
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
MoeScatterDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_scatter_dynamic_quant"]

# Dynamically load the localized SYCL extension
so_path = os.path.join(os.path.dirname(__file__), "moe_scatter_dynamic_quant_sycl.so")
if os.path.exists(so_path):
    spec = importlib.util.spec_from_file_location("moe_scatter_dynamic_quant_sycl", so_path)
    sycl_ext = importlib.util.module_from_spec(spec)
    sys.modules["moe_scatter_dynamic_quant_sycl"] = sycl_ext
    spec.loader.exec_module(sycl_ext)
else:
    sycl_ext = None

@ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "sycl_ext")
class SyclExtMoeScatterDynamicQuantOp(MoeScatterDynamicQuantOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["sycl_ext"]


    def vendor_impl(self):
        super().vendor_impl()
        import torch
        from xpu_perf.micro_perf.core.utils import OpTensorInfo
        
        self.vendor_local_selected_experts = [
            [
                expert_idx - self.experts_start_idx
                if self.experts_start_idx <= expert_idx < self.experts_end_idx
                else -1
                for expert_idx in token_experts
            ]
            for token_experts in self.all_select_experts
        ]

        self.input_tensor_info.update({
            "smooth_scale": OpTensorInfo(
                shape=[self.num_experts, self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
            "selected_experts_local": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.vendor_local_selected_experts, dtype=dtype, device=device)
            ),
            "smooth_scale_local": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "token_to_scatter_offset": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            )
        })
        
        from xpu_perf.micro_perf.core.utils import calc_tensor_size
        self.output_tensor_info.update({
            "scatter_tokens_offset": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1
            ),
            "experts_token_start": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset[:self.num_experts_per_rank], dtype=dtype, device=device)
            )
        })

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["hidden_states"]) / self.num_tokens * self.used_src_tokens + \
            calc_tensor_size(self.input_tensor_info["selected_experts_local"]) + \
            calc_tensor_size(self.input_tensor_info["moe_weights"]) + \
            calc_tensor_size(self.input_tensor_info["smooth_scale_local"]) * (self.dispatch_tokens / self.num_experts_per_rank / 4) + \
            calc_tensor_size(self.input_tensor_info["token_to_scatter_offset"])

        self.write_bytes = \
            calc_tensor_size(self.output_tensor_info["scatter_tokens"]) + \
            calc_tensor_size(self.output_tensor_info["scatter_per_token_scale"]) + \
            calc_tensor_size(self.output_tensor_info["scatter_tokens_offset"]) + \
            calc_tensor_size(self.output_tensor_info["experts_token_count"]) + \
            calc_tensor_size(self.output_tensor_info["experts_token_start"])

        self.io_bytes = self.read_bytes + self.write_bytes
        self._run_func = self.moe_scatter_dynamic_quant_run

    def vendor_parser(self):
        if self.dtype in ["bfloat16", "float16"] and self.dst_dtype in ["int8", "float8", "float8_e4m3", "float8_e4m3fn"]:
            pass
        else:
            raise ValueError(
                f"SyclExtMoeScatterDynamicQuantOp not support dtype {self.dtype} dst_dtype {self.dst_dtype}"
            )

    def moe_scatter_dynamic_quant_run(self, tensor_mapping):
        if sycl_ext is None:
            raise RuntimeError("moe_scatter_dynamic_quant_sycl.so not found. Did you run build.sh?")

        hidden_states = tensor_mapping["hidden_states"]
        selected_experts = tensor_mapping["selected_experts_local"]
        moe_weights = tensor_mapping["moe_weights"]
        token_to_scatter_offset = tensor_mapping["token_to_scatter_offset"]
        smooth_scale = tensor_mapping["smooth_scale_local"]

        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]
        scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_start = tensor_mapping["experts_token_start"]

        # Zero out workspace buffers for tracking states
        token_to_scatter_offset.zero_()
        experts_token_count.zero_()
        experts_token_start.zero_()

        num_shared_experts = self.args_dict.get("num_shared_experts", 0)

        sycl_ext.moe_scatter_dynamic_quant(
            selected_experts,
            moe_weights,
            token_to_scatter_offset,
            experts_token_count,
            experts_token_start,
            hidden_states,
            smooth_scale,
            scatter_tokens,
            scatter_per_token_scale,
            scatter_tokens_offset,
            num_shared_experts
        )

        return scatter_tokens
