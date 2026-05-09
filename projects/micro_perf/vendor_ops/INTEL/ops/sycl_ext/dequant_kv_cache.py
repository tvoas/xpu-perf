"""
SYCL extension provider for dequant_kv_cache.

Fuses int8→bf16 cast + scale multiply into a single kernel,
eliminating the intermediate bf16 tensor and saving ~2x memory traffic
compared to the two-step torch path (copy_ + mul_).
"""
from functools import partial
import torch
from xpu_perf.micro_perf.core.op import ProviderRegistry
import os
import pathlib
import importlib.util
import sys

# Get the torch vendor impl of dequant_kv_cache (already registered by the torch provider).
# If the torch provider's module is already in sys.modules, reuse it to avoid
# re-executing the file (which would overwrite the "torch" registration with a
# non-picklable module name).
_torch_mod_name = "xpu_perf_provider_torch.dequant_kv_cache"
if _torch_mod_name in sys.modules:
    DequantKVCacheOp = sys.modules[_torch_mod_name].DequantKVCacheOp
else:
    _torch_dequant_path = pathlib.Path(__file__).resolve().parent.parent / "torch" / "dequant_kv_cache.py"
    _spec_dq = importlib.util.spec_from_file_location(_torch_mod_name, _torch_dequant_path)
    _mod_dq = importlib.util.module_from_spec(_spec_dq)
    sys.modules[_torch_mod_name] = _mod_dq
    _spec_dq.loader.exec_module(_mod_dq)
    DequantKVCacheOp = _mod_dq.DequantKVCacheOp


# Load the compiled SYCL extension
_SYCL_SO = os.path.join(os.path.dirname(__file__), "dequant_kv_cache_sycl.so")

try:
    _spec = importlib.util.spec_from_file_location("dequant_kv_cache_sycl", _SYCL_SO)
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("dequant_kv_cache", "sycl_ext")
    class SYCLExtDequantKVCacheOp(DequantKVCacheOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def prepare(self):
            """Reuse base class tensor setup, override run func."""
            super().prepare()
            self._run_func = self.sycl_dequant_kv_cache_run

        def sycl_dequant_kv_cache_run(self, tensor_mapping):
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            dequant_k_cache = tensor_mapping["dequant_k_cache"]
            dequant_v_cache = tensor_mapping["dequant_v_cache"]
            k_scale = tensor_mapping["k_scale"]
            v_scale = tensor_mapping["v_scale"]

            bs = self.batch_size
            kv_len = self.kv_lens[0]
            uniform_kv_lens = all(b_kv_len == kv_len for b_kv_len in self.kv_lens)

            k_scale_bf16 = k_scale.to(torch.bfloat16)
            v_scale_bf16 = v_scale.to(torch.bfloat16)

            if self.cache_type == "paged":
                block_table = tensor_mapping["block_table"]
                kv_lens = tensor_mapping["kv_lens"]

                if self.dtype in ("float8", "float8_e4m3"):
                    _sycl_ext.dequant_kv_cache_fp8_paged(
                        k_cache, v_cache,
                        dequant_k_cache, dequant_v_cache,
                        k_scale_bf16, v_scale_bf16,
                        block_table, kv_lens,
                        bs,
                    )
                else:
                    _sycl_ext.dequant_kv_cache_paged(
                        k_cache, v_cache,
                        dequant_k_cache, dequant_v_cache,
                        k_scale_bf16, v_scale_bf16,
                        block_table, kv_lens,
                        bs,
                    )

                return dequant_k_cache, dequant_v_cache

            if self.cache_type != "linear" or not uniform_kv_lens:
                return super().dequant_kv_cache_run(tensor_mapping)

            if self.dtype in ("float8", "float8_e4m3"):
                _sycl_ext.dequant_kv_cache_fp8(
                    k_cache, v_cache,
                    dequant_k_cache, dequant_v_cache,
                    k_scale_bf16, v_scale_bf16,
                    bs, kv_len
                )
            else:
                _sycl_ext.dequant_kv_cache(
                    k_cache, v_cache,
                    dequant_k_cache, dequant_v_cache,
                    k_scale_bf16, v_scale_bf16,
                    bs, kv_len
                )

            return dequant_k_cache, dequant_v_cache

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL dequant_kv_cache extension: {e}")
