import os
import pathlib
import prettytable
from typing import Set, Optional, Callable, Dict, Any, List

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.engine import ComputeEngine, XCCLEngine
from xpu_perf.micro_perf.core.utils import logger


ENGINE_TYPE_MAPPING = {
    "ComputeEngine": ComputeEngine,
    "XCCLEngine": XCCLEngine
}


class XpuPerfServer:
    def __init__(self, args):
        self.started_engines = {}
        self.backend_instance = None


        self.backend_type = args.backend
        self.backend_name_list = args.backend_name_list
        self.backend_mod_list = args.backend_mod_list
        logger.info(f"Using backend {self.backend_type}")
        logger.info(f"Available backends: ")
        for backend_name in self.backend_name_list:
            logger.info(f"\t* {backend_name}: {self.backend_mod_list[backend_name]}")

        self.script_dir = args.script_dir
        self.vendor_ops_dir = self.script_dir.joinpath("vendor_ops")

        self.op_defs = args.op_defs
        self.vendor_ops = args.vendor_ops
        self.vendor_ops.append(self.vendor_ops_dir.joinpath(self.backend_type, "ops").absolute())

        logger.info(f"Using op_defs from: {self.op_defs}")
        logger.info("")

        logger.info(f"vendor_ops: {self.vendor_ops}")
        for op_path in self.vendor_ops:
            logger.info(f"\t* {op_path}")
        logger.info("")



        # create backend instance and load all avail ops
        try:
            backend_module = self.backend_mod_list[self.backend_type]
            backend_class = getattr(backend_module, "Backend" + self.backend_type)
            env_file = args.env
            if env_file is None:
                default_env_path = self.vendor_ops_dir.joinpath(self.backend_type, "env.json")
                if default_env_path.exists():
                    env_file = str(default_env_path)
            self.backend_instance = backend_class(
                backend=self.backend_type,
                env_file=env_file, 
                op_defs=self.op_defs,
                vendor_ops=self.vendor_ops
            )
        except Exception as e:
            logger.error(f"Failed to import backend {self.backend_type}: {e}")
            exit(1)

        # 获取系统基本信息
        common_pt = prettytable.PrettyTable()
        common_pt.field_names = ["attr", "value"]
        common_pt.align = "l"
        for attr, value in self.backend_instance.common_info.items():
            if attr == "numa_configs":
                continue
            else:
                common_pt.add_row([attr, value])    

        # 获取backend相关信息
        info_pt = prettytable.PrettyTable()
        info_pt.field_names = ["attr", "value"]
        info_pt.align = "l"
        for attr, value in self.backend_instance.backend_info.items():
            info_pt.add_row([attr, value])

        # 获取env相关信息
        env_pt = prettytable.PrettyTable()
        env_pt.field_names = ["env", "is_preset", "default_val", "final_val"]
        env_pt.align = "l"
        for attr in self.backend_instance.default_envs:
            if attr in self.backend_instance.override_envs:
                env_pt.add_row([attr, "True", self.backend_instance.default_envs[attr], os.environ[attr]])
            else:
                env_pt.add_row([attr, "False", self.backend_instance.default_envs[attr], os.environ[attr]])


        logger.info(f"Backend {self.backend_type} instance created.")
        logger.info(f"common info: \n{common_pt}")
        logger.info(f"backend info: \n{info_pt}")
        logger.info(f"env info: \n{env_pt}")



        # 解析 numa_config
        self.numa_config = args.numa
        if self.numa_config is None:
            self.numa_num = len(self.backend_instance.common_info["numa_configs"])
            self.numa_order = list(range(self.numa_num))    
        else:
            self.numa_num = len(self.numa_config.split(","))
            self.numa_order = [int(x) for x in self.numa_config.split(",")]
        self.device_mapping = list(range(self.backend_instance.backend_info["device_count"]))
        logger.info(f"use {self.numa_num} numa nodes, numa_order: {self.numa_order}, mapping to {self.device_mapping}")

        # 解析 node dist config
        self.node_world_size = args.node_world_size
        self.node_rank = args.node_rank
        self.all_numa_num = self.node_world_size * self.numa_num
        logger.info(f"node_world_size: {self.node_world_size}, node_rank: {self.node_rank}, all_numa_num: {self.all_numa_num}")

        # devices
        self.device = args.device
        if self.device is None:
            self.device_ids = self.device_mapping
        else:
            try:
                self.device_ids = [int(x) for x in self.device.split(",")]
            except ValueError:
                logger.error(f"Invalid device format: {self.device}, should be comma-separated integers")
                return 1
            for id in self.device_ids:
                if id not in self.device_mapping:
                    logger.error(f"Invalid device id: {id}, not in {self.device_mapping}")
                    return 1
        logger.info(f"using devices: {self.device_ids}")

        # 解析 serving config
        master_addr = args.master_addr
        server_port = args.server_port
        host_port = args.host_port
        device_port = args.device_port

        server_pt = prettytable.PrettyTable()
        server_pt.field_names = ["attr", "value", "note"]
        server_pt.align = "l"
        server_pt.add_row(["master_addr", master_addr, "ip address for serving and dist communication for gloo and xccl."])
        server_pt.add_row(["server_port", server_port, "port for serving requests."])
        server_pt.add_row(["host_port", host_port, "port for host communication."])
        server_pt.add_row(["device_port", device_port, "port for device communication."])
        logger.info(f"serving config: \n{server_pt}")

        for engine_name in ENGINE_TYPE_MAPPING:
            if engine_name == "XCCLEngine" and len(self.device_ids) * self.node_world_size <= 1:
                logger.info(f"skip starting {engine_name} since device num is {len(self.device_ids)} and node_world_size is {self.node_world_size}")
                continue
            
            args_dict = {
                "backend_instance": self.backend_instance,
                "node_world_size": self.node_world_size,
                "node_rank": self.node_rank,
                "master_addr": master_addr,
                "host_port": host_port,
                "device_port": device_port,
                "numa_num": self.numa_num,
                "numa_order": self.numa_order,
                "device_mapping": self.device_mapping,
                "device_ids": self.device_ids,
            }
            self.started_engines[engine_name] = ENGINE_TYPE_MAPPING[engine_name](**args_dict)

    def __del__(self):
        self.destroy()


    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()


    def create(self):
        self.destroy()        
        for engine_name, engine_instance in self.started_engines.items():
            engine_instance.start()
        if self.backend_instance:
            self.backend_instance.load_all_ops(print_provider=True)
            # 获取provider相关信息
            provider_pt = prettytable.PrettyTable()
            provider_pt.field_names = ["provider", "version"]
            provider_pt.align = "l"
            for provider, version in self.backend_instance.provider_info.items():
                provider_pt.add_row([provider, version])
            logger.info(f"provider info: \n{provider_pt}")


        

    def destroy(self):
        for engine_name, engine_instance in self.started_engines.items():
            engine_instance.stop()
        if self.backend_instance:
            try:
                self.backend_instance.clean_extra_files()
            except Exception:
                pass


    def get_info(self):
        info_dict = {
            "backend_type": self.backend_instance.backend_type, 
            "common": self.backend_instance.common_info, 
            "provider": self.backend_instance.provider_info, 
            "backend": self.backend_instance.backend_info, 
            "runtime": {
                "device_mapping": self.device_mapping, 
                "device_ids": self.device_ids, 
                "numa_num": self.numa_num, 
                "numa_order": self.numa_order, 
                "node_world_size": self.node_world_size, 
                "node_rank": self.node_rank, 
                "all_numa_num": self.all_numa_num, 
            }
        }
        return info_dict


    def bench(
        self,
        input_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        result_dict = {}

        """
        通信协议:
        {
            "type": "normal", 
            "data": {
                "flash_attention": [], 
                "gemm": []
            }
        }
        """
        try:
            input_type = input_dict["type"]
            data_dict = input_dict["data"]
            if input_type == "normal":
                result_dict = self.normal_bench(data_dict, progress_callback=progress_callback)
        except Exception as e:
            print(e)

        return result_dict

    def _grand_total_bench_steps(self, test_cases: Dict[str, List]) -> int:
        """与 normal_bench 中实际会推进的进度步数一致（含不支持算子按 case 计步）。"""
        total = 0
        for op_name, cases in test_cases.items():
            if op_name not in ProviderRegistry.OP_ENGINE_MAPPING:
                total += len(cases)
                continue
            if op_name not in self.backend_instance.op_mapping:
                total += len(cases)
                continue
            n_prov = len(self.backend_instance.op_mapping[op_name])
            total += len(cases) * n_prov
        return max(total, 1)

    def normal_bench(
        self,
        test_cases: Dict[str, List],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        {
            "flash_attention": [
                common_case_0
                common_case_1
            ], 
            "gemm": [
                common_case_0
                common_case_1
            ]
        }
        - key是op_name, value是列表, 包含多个待测试的公共配置参数
        - 每一个op在特定backend上可能有多个provider实现
        - 每一个op在特定provider实现上可能有多个实现方式, 比如高精、低精, 不同数据类型等,
        而这个参数是provider/backend特定的, 无法表示在workload上。
        """

        grand_total = self._grand_total_bench_steps(test_cases)
        completed = 0

        def emit_progress(extra: Dict[str, Any]):
            nonlocal completed
            completed += 1
            if progress_callback:
                pct = 100.0 * completed / grand_total if grand_total else 100.0
                progress_callback(
                    {
                        "event": "progress",
                        "completed": completed,
                        "total": grand_total,
                        "percent": round(pct, 2),
                        **extra,
                    }
                )

        # 将输入测试用例分发到不同的engine上
        engine_tasks = {}
        # 对等地创建results
        all_results = {}

        for op_name, cases in test_cases.items():
            # 如果op_name不被当前框架支持, 创建空结果
            if op_name not in ProviderRegistry.OP_ENGINE_MAPPING:
                logger.error(f"op_name: {op_name} not in OP_ENGINE_MAPPING")
                all_results[op_name] = [{} for _ in cases]
                for _ in cases:
                    emit_progress(
                        {
                            "op_name": op_name,
                            "op_total_steps": len(cases),
                            "note": "unsupported_op",
                        }
                    )
                continue
            engine_name = ProviderRegistry.OP_ENGINE_MAPPING[op_name]
            engine_tasks[engine_name] = engine_tasks.get(engine_name, {})
            engine_tasks[engine_name][op_name] = cases

        for engine_name, engine_test_cases in engine_tasks.items():
            # 如果engine_name没有在当前测试启动, 创建空结果
            if engine_name not in self.started_engines:
                logger.error(f"engine_name: {engine_name} not in self.started_engines")
                for op_name in engine_test_cases:
                    cases = engine_test_cases[op_name]
                    all_results[op_name] = [{} for _ in cases]
                    n_prov = len(self.backend_instance.op_mapping.get(op_name, {})) or 1
                    op_total_steps = len(cases) * n_prov
                    for _ in range(len(cases) * n_prov):
                        emit_progress(
                            {
                                "op_name": op_name,
                                "op_total_steps": op_total_steps,
                                "note": "engine_not_started",
                            }
                        )
                continue

            num_ops = len(engine_test_cases)
            num_cases = sum([len(cases) for cases in engine_test_cases.values()])

            # 直接dispatch到对应的engine上
            logger.info(f"dispatch {num_ops} ops, {num_cases} cases to {engine_name}")

            def dispatch_progress(meta: Dict[str, Any]):
                emit_progress(
                    {
                        "engine": meta.get("engine", engine_name),
                        "op_name": meta.get("op_name"),
                        "op_provider": meta.get("op_provider"),
                        "case_index": meta.get("case_index"),
                        "cases_for_op": meta.get("cases_for_op"),
                        "op_total_steps": meta.get("op_total_steps"),
                    }
                )

            cur_results = self.started_engines[engine_name].dispatch(
                engine_test_cases,
                progress_callback=dispatch_progress,
                engine_name=engine_name,
            )
            all_results.update(cur_results)

        return all_results

