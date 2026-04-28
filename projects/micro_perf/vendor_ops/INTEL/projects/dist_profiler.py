import os
import pathlib
import shutil
import json
import time
from datetime import timedelta

import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])


profiling_dir = pathlib.Path.cwd().joinpath(f"profiling_{local_rank}")
trace_file = profiling_dir.joinpath(f"trace.json")


if profiling_dir.exists():
    shutil.rmtree(profiling_dir)
profiling_dir.mkdir(parents=True)


torch.xpu.set_device(local_rank)


dist.init_process_group(
    backend="ccl", 
    world_size=world_size, 
    rank=rank, 
    timeout=timedelta(seconds=1800)
)

tensor = torch.ones(1, dtype=torch.float16, device="xpu")
dist.all_reduce(tensor)
print(f"rank {rank}, {tensor}")


warmup_iters = 10
test_iters = 10
tensor = torch.ones([256 * 1024, 1024], dtype=torch.float32, device="xpu")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.XPU], 
    schedule=torch.profiler.schedule(
        wait=0, 
        warmup=warmup_iters, 
        active=test_iters, 
        repeat=1
    ), 
    on_trace_ready=lambda prof: prof.export_chrome_trace(f"{trace_file}")
) as prof:
    for _ in range(warmup_iters + test_iters):
        work = dist.all_reduce(tensor, async_op=True)
        work.wait()
        torch.xpu.synchronize()
        prof.step()

dist.destroy_process_group()


profiling_dict = json.loads(trace_file.read_text())

event_mapping = {}
for event in profiling_dict["traceEvents"]:
    if event.get("cat", "") == "kernel":
        kernel_name = event["name"]
        if kernel_name not in event_mapping:
            event_mapping[kernel_name] = []
        event_mapping[kernel_name].append(event)

for kernel_name, event_list in event_mapping.items():
    event_list.sort(key=lambda x: x["ts"])


print("")

for kernel_name, event_list in event_mapping.items():
    latency_list = [event["dur"] for event in event_list]
    print(f"{local_rank}, {kernel_name}: {latency_list}")

print("")