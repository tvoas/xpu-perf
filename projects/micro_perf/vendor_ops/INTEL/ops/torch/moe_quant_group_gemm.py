import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype
BaseMoeQuantGroupGemmOp = ProviderRegistry.BASE_IMPL_MAPPING["moe_quant_group_gemm"]


@ProviderRegistry.register_vendor_impl("moe_quant_group_gemm", "torch")
class MoeQuantGroupGemmTorchOp(BaseMoeQuantGroupGemmOp):
    """Override vendor_parser to accept torch-supported dtype combinations."""

    SUPPORTED_DTYPE_COMBINATIONS = {
        ("int8", "int8", "int8", "bfloat16"),
        ("bfloat16", "bfloat16", "bfloat16", "bfloat16"),
    }

    def vendor_parser(self):
        current_combination = (
            self.dtype,
            self.w_dtype,
            self.compute_dtype,
            self.dst_dtype,
        )
        if current_combination in self.SUPPORTED_DTYPE_COMBINATIONS:
            return

        raise ValueError(
            "MoeQuantGroupGemmTorchOp supports dtype/w_dtype/compute_dtype/dst_dtype "
            f"in {sorted(self.SUPPORTED_DTYPE_COMBINATIONS)}, but got {current_combination}"
        )

    def vendor_impl(self):
        current_combination = (
            self.dtype,
            self.w_dtype,
            self.compute_dtype,
            self.dst_dtype,
        )
        if current_combination != ("bfloat16", "bfloat16", "bfloat16", "bfloat16"):
            return super().vendor_impl()

        self.torch_dtype = get_torch_dtype(self.dtype)
        self.w_torch_dtype = get_torch_dtype(self.w_dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros,
            ),
            "experts_weight": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.new_hidden_size, self.hidden_size],
                dtype=self.w_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros,
            ),
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank],
                dtype=torch.int32,
                device="cpu",
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device
                ),
            ),
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.dispatch_tokens, self.new_hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        self.active_experts = sum(1 for token_count in self.expert_dispatch_token_count if token_count > 0)

        self.input_tensor_size = sum(
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        )
        self.output_tensor_size = sum(
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        token_bytes = calc_tensor_size(self.input_tensor_info["scatter_tokens"])
        token_count_bytes = calc_tensor_size(self.input_tensor_info["experts_token_count"])
        weight_dtype_size = self.input_tensor_info["experts_weight"].dtype.itemsize
        active_weight_bytes = (
            self.active_experts * self.new_hidden_size * self.hidden_size * weight_dtype_size
        )

        self.read_bytes = token_bytes + token_count_bytes + active_weight_bytes
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        current_combination = (
            self.dtype,
            self.w_dtype,
            self.compute_dtype,
            self.dst_dtype,
        )
        if current_combination != ("bfloat16", "bfloat16", "bfloat16", "bfloat16"):
            return super().vendor_impl_run(tensor_mapping)

        scatter_tokens = tensor_mapping["scatter_tokens"]
        experts_weight = tensor_mapping["experts_weight"]
        experts_token_count = tensor_mapping["experts_token_count"]
        y = tensor_mapping["y"]

        token_counts = experts_token_count.tolist()
        token_start = 0

        for expert_idx in range(self.num_experts_per_rank):
            token_count = token_counts[expert_idx]
            if token_count == 0:
                continue

            token_end = token_start + token_count

            cur_tokens = scatter_tokens[token_start:token_end]
            cur_weight = experts_weight[expert_idx].transpose(0, 1)

            torch.matmul(cur_tokens, cur_weight, out=y[token_start:token_end])
            token_start = token_end

        return y

