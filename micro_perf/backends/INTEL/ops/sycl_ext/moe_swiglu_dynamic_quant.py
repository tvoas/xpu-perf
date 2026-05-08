import os
import sys
import importlib.util
import torch
from functools import partial

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeSwigluDynamicQuantOp

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
        self._run_func = self.moe_swiglu_dynamic_quant_run

    def moe_swiglu_dynamic_quant_run(self, tensor_mapping):
        if sycl_ext is None:
            raise RuntimeError("moe_swiglu_dynamic_quant_sycl.so not found. Did you run build.sh?")

        scatter_tokens = tensor_mapping["scatter_tokens"]
        smooth_scale = tensor_mapping["smooth_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_start = tensor_mapping["experts_token_start"]
        scatter_expert_ids = tensor_mapping["scatter_expert_ids"]

        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        total_experts_num = experts_token_count.size(0)
        max_token_num = int(experts_token_count.max().item())

        sycl_ext.moe_swiglu_dynamic_quant(
            scatter_tokens,
            smooth_scale,
            experts_token_count,
            experts_token_start,
            scatter_expert_ids,
            quant_tokens,
            per_token_scale,
            total_experts_num,
            max_token_num
        )

        return quant_tokens, per_token_scale