import torch
from functools import partial

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeScatterDynamicQuantOp, MoeSwigluDynamicQuantOp

try:
    import vllm_xpu_kernels._C  # noqa: F401
    import vllm_xpu_kernels._moe_C

    # -------------------------------------------------------------
    # 1. MOE Scatter Dynamic Quant
    # -------------------------------------------------------------
    @ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeScatterDynamicQuantOp(MoeScatterDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )
            self._run_func = self.moe_scatter_dynamic_quant_run

        def moe_scatter_dynamic_quant_run(self, tensor_mapping):
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

            token_to_scatter_offset.zero_()
            experts_token_count.zero_()
            experts_token_start.zero_()

            num_shared_experts = self.args_dict.get("num_shared_experts", 0)

            torch.ops._moe_C.moe_scatter_dynamic_quant(
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

    # -------------------------------------------------------------
    # 2. MOE SwiGLU Dynamic Quant
    # -------------------------------------------------------------
    @ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeSwigluDynamicQuantOp(MoeSwigluDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )
            self._run_func = self.moe_swiglu_dynamic_quant_run

        def moe_swiglu_dynamic_quant_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            smooth_scale = tensor_mapping["smooth_scale"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            # C++ backend takes total experts instead of experts per rank
            total_experts_num = experts_token_count.size(0)
            max_token_num = int(experts_token_count.max().item())

            torch.ops._moe_C.moe_swiglu_dynamic_quant(
                scatter_tokens,
                smooth_scale,
                experts_token_count,
                experts_token_start,
                quant_tokens,
                per_token_scale,
                total_experts_num,
                max_token_num
            )

            return quant_tokens, per_token_scale

except ImportError:
    pass
