from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.utils import smooth_per_token_dynamic_quant

BaseMoeScatterDynamicQuantOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_scatter_dynamic_quant"]


@ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "torch")
class MoeScatterDynamicQuantTorchOp(BaseMoeScatterDynamicQuantOp):

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]

        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = self.expert_dispatch_token_offset[expert_idx]
            cur_token_end = cur_token_start + self.expert_dispatch_token_count[expert_idx]
            if self.expert_dispatch_token_count[expert_idx] == 0:
                continue

            src_token_indices = scatter_token_id[cur_token_start:cur_token_end]

            scatter_tokens[cur_token_start:cur_token_end], \
            scatter_per_token_scale[cur_token_start:cur_token_end] = \
                smooth_per_token_dynamic_quant(
                    hidden_states[src_token_indices],
                    experts_smooth_scale[expert_idx],
                    dst_torch_dtype=self.dst_torch_dtype
                )

        return scatter_tokens, scatter_per_token_scale, \
               scatter_token_id, scatter_token_weight, \
               experts_token_count, experts_token_offset
