import os
import sys
import importlib.util
import torch
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
MoeScatterDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_scatter_dynamic_quant"]

# Dynamically load the localized SYCL extension.
#
# The compiled .so contains the pybind11 entry point implemented in the C++ SYCL
# file. This Python module exists to register that compiled kernel as one vendor
# backend inside the xpu-perf microbenchmark framework.
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
        # Let the base benchmark op create its generic tensor definitions first.
        # This override then replaces or augments them with the metadata/workspace
        # tensors required by the fused scatter + quantization kernel.
        super().vendor_impl()
        import torch
        from xpu_perf.micro_perf.core.utils import OpTensorInfo
        
        # Convert global expert ids into the expert-id space local to this rank.
        # Any expert outside the local shard becomes -1, which the kernel treats
        # as an invalid route and skips.
        self.vendor_local_selected_experts = [
            [
                expert_idx - self.experts_start_idx
                if self.experts_start_idx <= expert_idx < self.experts_end_idx
                else -1
                for expert_idx in token_experts
            ]
            for token_experts in self.all_select_experts
        ]

        # Extra tensors added by this vendor implementation:
        #   smooth_scale_local
        #       only the smooth-scale rows for experts owned by this rank
        #   selected_experts_local
        #       local expert ids or -1 for nonlocal experts
        #   token_to_scatter_offset
        #       workspace written by the routing phase inside the kernel
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
        # Additional outputs produced by the fused kernel:
        #   scatter_tokens_offset: original source token id for each scattered row
        #   experts_token_start: prefix-sum start of each expert segment
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

        # These bandwidth estimates are benchmark accounting numbers rather than
        # byte-exact hardware counters. They model the logical payload that the
        # op conceptually reads and writes.
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
        # Restrict benchmark configurations to the dtype combinations implemented
        # by the C++ dispatch layer.
        if self.dtype in ["bfloat16", "float16"] and self.dst_dtype in ["int8", "float8", "float8_e4m3", "float8_e4m3fn"]:
            pass
        else:
            raise ValueError(
                f"SyclExtMoeScatterDynamicQuantOp not support dtype {self.dtype} dst_dtype {self.dst_dtype}"
            )

    def moe_scatter_dynamic_quant_run(self, tensor_mapping):
        if sycl_ext is None:
            raise RuntimeError("moe_scatter_dynamic_quant_sycl.so not found. Did you run build.sh?")

        # Inputs consumed by the fused kernel.
        # selected_experts_local already contains rank-local expert ids, so the
        # device kernel can use them directly as dense indices into local expert
        # metadata arrays.
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

        # Zero the workspaces before launch.
        # The kernel fills these tensors in place as part of the routing stage,
        # so each benchmark iteration must start from a clean state.
        token_to_scatter_offset.zero_()
        experts_token_count.zero_()
        experts_token_start.zero_()

        # Shared experts, if any, have already been normalized into the routing
        # table. The scalar is still forwarded to keep parity with the generic op
        # interface and with the underlying pybind signature.
        num_shared_experts = self.args_dict.get("num_shared_experts", 0)

        # Enter the compiled extension. The C++ side performs both the routing
        # metadata construction and the actual scatter + quantization work.
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

        # The benchmark currently treats scatter_tokens as the primary returned
        # payload, even though the kernel also writes several side-output tensors.
        return scatter_tokens
