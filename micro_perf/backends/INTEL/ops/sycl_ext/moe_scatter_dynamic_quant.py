import os
import sys
import importlib.util
import torch
from functools import partial

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeScatterDynamicQuantOp

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
        self._run_func = self.moe_scatter_dynamic_quant_run

    def moe_scatter_dynamic_quant_run(self, tensor_mapping):
        if sycl_ext is None:
            raise RuntimeError("moe_scatter_dynamic_quant_sycl.so not found. Did you run build.sh?")

        hidden_states = tensor_mapping["hidden_states"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        token_to_scatter_offset = tensor_mapping["token_to_scatter_offset"]
        smooth_scale = tensor_mapping["smooth_scale"]

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