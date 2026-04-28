import torch
import pathlib
import shutil
import json


profiling_dir = pathlib.Path.cwd().joinpath("profiling")


torch.xpu.set_device(0)


a = torch.ones([4096, 4096], dtype=torch.float16, device="xpu")
b = torch.ones([4096, 4096], dtype=torch.float16, device="xpu")
c = torch.ones([4096, 4096], dtype=torch.float16, device="xpu")
def gemm_func():
    torch.matmul(a, b, out=c)


def add_func():
    torch.add(a, b, out=c)


warmup_iters = 10
test_iters = 10


if profiling_dir.exists():
    shutil.rmtree(profiling_dir)
profiling_dir.mkdir(parents=True)


with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.XPU], 
    schedule=torch.profiler.schedule(
        wait=0, 
        warmup=warmup_iters, 
        active=test_iters, 
        repeat=1
    )
) as prof:
    for _ in range(warmup_iters + test_iters):
        add_func()
        torch.xpu.synchronize()
        prof.step()

torch.profiler.tensorboard_trace_handler(f"{profiling_dir}")(prof)

json_file = list(profiling_dir.glob("*.json"))[0]

profiling_dict = json.loads(json_file.read_text())

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
    print(f"{kernel_name}: {latency_list}")

print("")