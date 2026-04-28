import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "vllm_xpu_kernels"

try:
    import vllm_xpu_kernels._C
    ProviderRegistry.register_provider_info("vllm_xpu_kernels", {
        "vllm_xpu_kernels": importlib.metadata.version("vllm-xpu-kernels"),
    })
except Exception:
    pass
