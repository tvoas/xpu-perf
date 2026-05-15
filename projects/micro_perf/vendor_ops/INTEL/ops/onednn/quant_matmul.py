"""oneDNN provider for quant_matmul.

Calls the pybind11 C++ extension (quant_matmul_onednn.so) which invokes the
oneDNN matmul primitive on PyTorch XPU tensors via SYCL interop (zero-copy).

API (also usable standalone):
    import quant_matmul_onednn
    quant_matmul_onednn.quant_matmul(hidden_states, per_token_scale,
                                      weight, weight_scale, out)
"""

import os
import pathlib
import importlib.util
import torch
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry, BasicOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size

QuantMatmulOp = ProviderRegistry.BASE_IMPL_MAPPING["quant_matmul"]

_OP_DIR = pathlib.Path(__file__).resolve().parent
_SO_PATH = str(_OP_DIR / "quant_matmul_onednn.so")

try:
    if not os.path.isfile(_SO_PATH):
        print(
            f"[WARNING] {_SO_PATH} not found. "
            f"Run 'bash build.sh' in {_OP_DIR} first. "
            f"onednn quant_matmul provider will NOT be available."
        )
        raise FileNotFoundError(_SO_PATH)

    _spec = importlib.util.spec_from_file_location(
        "quant_matmul_onednn", _SO_PATH
    )
    _ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ext)

    @ProviderRegistry.register_vendor_impl("quant_matmul", "onednn")
    class OneDNNQuantMatmulOp(BasicOp):
        """W8A8 quantised matmul via oneDNN (PyTorch tensor input, framework timing)."""

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict["dtype"]
            if self.dtype not in ["int8"]:
                raise NotImplementedError

            self.sp_size = self.args_dict.get("sp_size", 1)
            self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
            self.hidden_size = self.args_dict["hidden_size"]
            self.new_hidden_size = self.args_dict["new_hidden_size"]
            self.trans_w = self.args_dict.get("trans_w", False)

            w_shape = (
                [self.hidden_size, self.new_hidden_size]
                if self.trans_w
                else [self.new_hidden_size, self.hidden_size]
            )

            dst_dtype_str = self.args_dict.get("dst_dtype", "bfloat16")
            dst_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.dst_torch_dtype = dst_dtype_map.get(dst_dtype_str, torch.bfloat16)

            dev = self.backend.get_torch_device_name()

            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=torch.int8, device=dev,
                ),
                "weight": OpTensorInfo(
                    shape=w_shape,
                    dtype=torch.int8, device=dev,
                ),
                "per_token_scale": OpTensorInfo(
                    shape=[self.num_tokens],
                    dtype=torch.float32, device=dev,
                ),
                "weight_scale": OpTensorInfo(
                    shape=[self.new_hidden_size],
                    dtype=torch.float32, device=dev,
                ),
                "zero_points": OpTensorInfo(
                    shape=[1],
                    dtype=torch.int8, device=dev,
                ),
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.num_tokens, self.new_hidden_size],
                    dtype=self.dst_torch_dtype, device=dev,
                ),
            }

            self.input_tensor_size = sum(
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            )
            self.output_tensor_size = sum(
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0
            self.calc_flops = (
                2 * self.num_tokens * self.hidden_size * self.new_hidden_size
            )

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

            self._run_func = self.quant_matmul_run

        def quant_matmul_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            weight = tensor_mapping["weight"]
            per_token_scale = tensor_mapping["per_token_scale"]
            weight_scale = tensor_mapping["weight_scale"]
            out = tensor_mapping["out"]

            # C++ ext accepts weight as [N, K] — no transpose needed
            _ext.quant_matmul(
                hidden_states,
                per_token_scale,
                weight,
                weight_scale,
                out,
            )


except Exception as e:
    print(f"[OneDNNQuantMatmulOp] Failed to register: {e}")
    pass

