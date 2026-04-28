import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "ipex"

try:
    ProviderRegistry.register_provider_info("ipex", {
        "intel_extension_for_pytorch": importlib.metadata.version("intel-extension-for-pytorch"),
    })
except Exception:
    pass
