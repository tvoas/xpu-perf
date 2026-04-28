
def check_intel_env():
    import torch
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        raise EnvironmentError("Intel XPU is not available. Please check your GPU environment.")
    else:
        print(f"Intel XPU is available. Found {torch.xpu.device_count()} XPU device(s).")

check_intel_env()

from .backend_intel import BackendINTEL
