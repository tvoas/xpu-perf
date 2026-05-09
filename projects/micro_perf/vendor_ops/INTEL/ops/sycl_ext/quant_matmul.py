import os
import pathlib
import re
import sys
import tempfile
import importlib.util
import ctypes

from xpu_perf.micro_perf.core.op import ProviderRegistry

QuantMatmulOp = ProviderRegistry.BASE_IMPL_MAPPING["quant_matmul"]


def _flush_all_stdio():
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


_OP_DIR = pathlib.Path(__file__).resolve().parent
SYCL_EXT_SO = str(_OP_DIR / "quant_matmul_sycl.so")


try:
    if not os.path.isfile(SYCL_EXT_SO):
        print(
            f"[WARNING] sycl_ext shared object not found: {SYCL_EXT_SO}. "
            "sycl_tla quant_matmul provider will NOT be available."
        )
        raise FileNotFoundError(SYCL_EXT_SO)

    _spec = importlib.util.spec_from_file_location(
        "quant_matmul_sycl", SYCL_EXT_SO
    )
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)


    @ProviderRegistry.register_vendor_impl("quant_matmul", "sycl_tla")
    class SyclTlaQuantMatmulOp(QuantMatmulOp):
        def vendor_parser(self):
            if not (
                self.dtype == "int8"
                and self.w_dtype == "int8"
                and self.compute_dtype == "int8"
                and self.dst_dtype == "bfloat16"
            ):
                raise ValueError(
                    "sycl_tla only supports "
                    "int8 x int8 -> bfloat16 (compute=int8), got "
                    f"dtype={self.dtype}, w_dtype={self.w_dtype}, "
                    f"compute_dtype={self.compute_dtype}, dst_dtype={self.dst_dtype}"
                )

        def vendor_impl(self):
            # 让 base impl 计算 calc_flops / io_bytes / read_bytes / write_bytes,
            # 这样 summary() 报告里的 TFLOPS / mem_bw 才是非零.
            super().vendor_impl()

            # 跑 SYCL 扩展, 直接拿 latency, 不再走 micro_perf 的 core_run / 张量分配.
            self._provider = "sycl_tla"
            self._sycl_tla_result = self._run_sycl_ext()

            # micro_perf still wants per-iter timing hooks; the actual GPU work was
            # already done inside the SYCL extension call above, and we re-use the
            # parsed latency in summary().
            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = (
                lambda instance_num: [{} for _ in range(max(instance_num, 1))]
            )

        def _capture_extension_output(self, func, *args, **kwargs):
            with tempfile.TemporaryFile(mode="w+b") as temp_file:
                stdout_fd = os.dup(sys.stdout.fileno())
                stderr_fd = os.dup(sys.stderr.fileno())
                try:
                    _flush_all_stdio()
                    os.dup2(temp_file.fileno(), sys.stdout.fileno())
                    os.dup2(temp_file.fileno(), sys.stderr.fileno())
                    result = func(*args, **kwargs)
                    _flush_all_stdio()
                finally:
                    _flush_all_stdio()
                    os.dup2(stdout_fd, sys.stdout.fileno())
                    os.dup2(stderr_fd, sys.stderr.fileno())
                    os.close(stdout_fd)
                    os.close(stderr_fd)

                temp_file.seek(0)
                output_text = temp_file.read().decode("utf-8", errors="replace")
                return result, output_text

        def _parse_sycl_tla_output(self, output_text):
            # Example line:
            # "Cutlass W8A8 Performance:    [ 53.26] TOp/s     0.0441 ms     353.3 GB/s"
            perf_match = re.search(
                r"Cutlass W8A8 Performance:\s*\[\s*([\d\.eE+-]+)\s*\]\s*TOp/s"
                r"\s+([\d\.eE+-]+)\s*ms\s+([\d\.eE+-]+)\s*GB/s",
                output_text,
            )
            if perf_match is None:
                raise RuntimeError(
                    "Failed to parse quant_matmul output:\n"
                    f"{output_text}"
                )

            return {
                "tops": float(perf_match.group(1)),
                "latency_ms": float(perf_match.group(2)),
                "gbps": float(perf_match.group(3)),
                "raw_output": output_text,
            }

        def _run_sycl_ext(self):
            m = self.num_tokens
            n = self.new_hidden_size
            k = self.hidden_size
            with_bias = 1 if self.has_bias else 0

            command = (
                f"quant_matmul_sycl.run_quant_matmul("
                f"m={m}, n={n}, k={k}, l=1, with_bias={with_bias}, "
                f"iterations=50, verify=0)"
            )
            print(f"[SyclTlaQuantMatmulOp] Running: {command}")

            retcode, output_text = self._capture_extension_output(
                _sycl_ext.run_quant_matmul,
                m=m,
                n=n,
                k=k,
                l=1,
                with_bias=with_bias,
                iterations=50,
                verify=0,
            )

            if retcode != 0:
                raise RuntimeError(
                    f"sycl_ext call failed (rc={retcode}):\n{output_text}"
                )

            print(f"[SyclTlaQuantMatmulOp] Output:\n{output_text}")

            parsed_result = self._parse_sycl_tla_output(output_text)
            return {
                "latency_ms": parsed_result["latency_ms"],
                "tops": parsed_result["tops"],
                "gbps": parsed_result["gbps"],
                "raw_output": parsed_result["raw_output"],
                "command": command,
            }

        def summary(self, latency_us, kernel_mapping={}):
            if self._sycl_tla_result:
                latency_us = self._sycl_tla_result["latency_ms"] * 1000.0
                kernel_mapping = dict(kernel_mapping)
                kernel_mapping["tops"] = self._sycl_tla_result["tops"]
                kernel_mapping["gbps"] = self._sycl_tla_result["gbps"]
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[SyclTlaQuantMatmulOp] Failed to register: {e}")
    pass
