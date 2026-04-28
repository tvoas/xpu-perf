import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[4]))

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeSwigluDynamicQuantOp

try:
    torch.ops.torch_ipex.moe_swiglu_dynamic_quant

    @ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "ipex")
    class MoeSwigluDynamicQuantIpexOp(MoeSwigluDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["ipex"]

            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )
            self._run_func = self.moe_swiglu_dynamic_quant_run


        def moe_swiglu_dynamic_quant_run(self, tensor_mapping): 
            # get pre-allocated input tensors
            scatter_tokens = tensor_mapping["scatter_tokens"]
            smooth_scale = tensor_mapping["smooth_scale"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            # get per-allocated output tensors
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            # IPEX requires experts_per_rank (calculated from config) and max token count dynamically
            experts_per_rank = self.args_dict.get("num_experts", experts_token_count.size(0)) // self.args_dict.get("ep_size", 1)
            max_token_num = int(experts_token_count.max().item())

            torch.ops.torch_ipex.moe_swiglu_dynamic_quant(
                scatter_tokens,
                smooth_scale,
                experts_token_count,
                experts_token_start,
                quant_tokens,
                per_token_scale,
                experts_per_rank,
                max_token_num
            )

            return quant_tokens, per_token_scale

except Exception:
    pass
