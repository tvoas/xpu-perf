import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[4]))

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeScatterDynamicQuantOp

try:
    torch.ops.torch_ipex.moe_scatter_dynamic_quant

    @ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "ipex")
    class MoeScatterDynamicQuantIpexOp(MoeScatterDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["ipex"]
            self._run_func = self.moe_scatter_dynamic_quant_run

        def moe_scatter_dynamic_quant_run(self, tensor_mapping):
            # get pre-allocated input tensors
            hidden_states = tensor_mapping["hidden_states"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]
            token_to_scatter_offset = tensor_mapping["token_to_scatter_offset"]
            smooth_scale = tensor_mapping["smooth_scale"]

            # get pre-allocated output tensors
            scatter_tokens = tensor_mapping["scatter_tokens"]
            scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]
            scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            token_to_scatter_offset.zero_()
            experts_token_count.zero_()
            experts_token_start.zero_()

            num_shared_experts = self.args_dict.get("num_shared_experts", 0)

            torch.ops.torch_ipex.moe_scatter_dynamic_quant(
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

except Exception:
    pass
