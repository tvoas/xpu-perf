import os
import sys
import csv
import json
import copy
import argparse
import pathlib
import logging
import jsonlines
import itertools
import prettytable
from typing import List, Dict, Any

import importlib.util
import importlib
import pkgutil




logger = logging.getLogger("bytemlperf_micro_perf")
def setup_logger(loglevel: str):
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel.upper())
    logger.propagate = False



"""
task_name,arg1,arg2,arg3
name1, value1, value2, value3
"""
def parse_csv_file(csv_file : pathlib.Path):
    task_dict : Dict[str, List[Dict[str, Any]]] = {}

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        
        if "task_name" not in reader.fieldnames:
            logger.error(f"CSV file {csv_file} must contain task_name column")
            sys.exit(1)

        for data in reader:
            task_name = data.pop("task_name")

            if task_name not in task_dict:
                task_dict[task_name] = []


            task_dict[task_name].append(data)

    return task_dict


"""
{task_name}.json
{
    "cases": [
        {
            "arg1": ["value1"],
            "arg2": ["value2"],
            "arg3": ["value3"],
        }
    ]
}
--> task_name, arg1, arg2, arg3
--> name1, value1, value2, value3


{any_name}.json
{
    "name1": [
        {
            "arg1": ["value1"],
            "arg2": ["value2"],
            "arg3": ["value3"],
        }
    ], 
    "name2": [
        {
            "arg1": ["value1"],
            "arg2": ["value2"],
            "arg3": ["value3"],
        }
    ]
}
"""

def get_cartesian_product(test_cases: List[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
    if not test_cases:
        return []

    return_datas = []
    for argument_case in test_cases:
        if "batch" in argument_case["arg_type"]:
            return_datas.append(argument_case)
            continue
            
        keys = list(argument_case.keys())
        values = list(argument_case.values())
        for i, value in enumerate(values):
            if not isinstance(value, list):
                values[i] = [value]
        total_cases = list(itertools.product(*values))
        for case in total_cases:
            case_dict = dict(zip(keys, case))

            added_key_value = {}
            removed_key = []
            for key in case_dict:
                if "." in key:
                    split_keys = key.split(".")
                    for i, split_key in enumerate(split_keys):
                        added_key_value[split_key] = case_dict[key][i]
                    removed_key.append(key)
            for key in removed_key:
                del case_dict[key]
            case_dict.update(added_key_value)
            return_datas.append(case_dict)
    return return_datas


def parse_json_file(json_file : pathlib.Path):
    task_dict : Dict[str, List[Dict[str, Any]]] = {}

    parsed_data = json.loads(json_file.read_text())

    if "cases" in parsed_data:
        task_name = json_file.stem
        task_dict[task_name] = get_cartesian_product(parsed_data["cases"])
    else:
        for task_name, test_cases in parsed_data.items():
            task_dict[task_name] = get_cartesian_product(test_cases)

    return task_dict




def existing_dir_path(path_str):
    """
    检查路径是否存在且为目录，如果是则返回 Path 对象，否则抛出 ArgumentTypeError
    """
    path = pathlib.Path(path_str)
    
    # 检查路径是否存在
    if not path.exists():
        raise argparse.ArgumentTypeError(f"路径不存在: '{path_str}'")
        
    # 检查路径是否为文件夹
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"路径存在，但不是一个文件夹: '{path_str}'")
        
    return path


def valid_file(model_path_str: str, required_suffix: str) -> pathlib.Path:
    if model_path_str is None:
        return None

    model_path = pathlib.Path(model_path_str).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"File {model_path} not found!")
    if not model_path.is_file():
        raise FileNotFoundError(f"File {model_path} is not a file!")
    if not model_path.suffix == required_suffix:
        raise ValueError(f"File {model_path} is not a json file!")
    return model_path


def get_submodules(package_name):
    """
    获取指定包的直接子模块信息。
    
    参数:
        package_name (str): 父包的绝对导入路径，例如 "xpu_perf.backend"
        
    返回:
        tuple: (子模块名列表, {子模块名: 模块对象})
    """
    names = []
    modules_dict = {}

    try:
        # 1. 导入顶层父包
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"❌ 导入父包 '{package_name}' 失败: {e}")
        return names, modules_dict

    # 2. 确保它是一个包（有 __path__ 属性的才是包，单文件模块没有）
    if not hasattr(package, "__path__"):
        print(f"⚠️ '{package_name}' 不是一个包目录，无法获取子模块。")
        return names, modules_dict

    # 3. 使用 iter_modules 浅层遍历当前包目录
    for module_info in pkgutil.iter_modules(package.__path__):
        sub_name = module_info.name
        full_name = f"{package_name}.{sub_name}"
        
        try:
            # 动态加载子模块
            mod = importlib.import_module(full_name)
            names.append(sub_name)
            modules_dict[sub_name] = mod
        except Exception as e:
            # 使用 Exception 而不是 ImportError，防止子模块内部代码报错导致整个扫描崩溃
            print(f"⚠️ 无法加载子模块 '{full_name}': {e}")

    return names, modules_dict


def load_all_python_files(package_name):
    """
    递归扫描并加载指定包及其所有子层级下的全部 Python 文件。
    
    参数:
        package_name (str): 顶层包名，例如 "xpu_perf"
        
    返回:
        list: 成功加载的模块对象列表（可选）
    """
    loaded_modules = []

    try:
        # 1. 先加载顶层包
        package = importlib.import_module(package_name)
        loaded_modules.append(package)
    except ImportError as e:
        print(f"❌ 导入顶层包 '{package_name}' 失败: {e}")
        return loaded_modules

    if not hasattr(package, "__path__"):
        return loaded_modules

    # 2. 关键：构造前缀。walk_packages 需要前缀来拼接完整的绝对导入名
    prefix = package.__name__ + "."

    # 3. walk_packages 会深入所有包含 __init__.py 的子孙文件夹
    for module_info in pkgutil.walk_packages(package.__path__, prefix):
        full_name = module_info.name
        
        try:
            # 动态执行导入（如果是已经被导入过的，importlib 会直接从 sys.modules 返回缓存，不会重复执行）
            mod = importlib.import_module(full_name)
            loaded_modules.append(mod)
            # print(f"✅ 成功加载: {full_name}")  # 调试用
        except Exception as e:
            # 在深层遍历时，极容易遇到某些模块缺少特定第三方依赖的情况
            # 捕获异常防止中断整个扫描过程
            print(f"⚠️ 递归加载跳过 '{full_name}': {e}")

    return loaded_modules



def parse_vendor_ops(package_name: str, package_init_path: str):
    """
    递归扫描当前包下所有 .py 文件，并通过相对路径动态导入
    :param package_name: 当前包的名称（固定传 __name__）
    :param package_init_path: 当前包 __init__.py 的路径（固定传 __file__）
    """
    # 1. 获取当前包的根目录（__init__.py 所在文件夹）
    package_dir = pathlib.Path(package_init_path).parent
    # 2. 递归查找所有 .py 文件（包含子目录）
    py_files = package_dir.rglob("*.py")

    imported_modules = []
    # 遍历所有Python文件
    for py_file in py_files:
        # 跳过所有 __init__.py，避免循环导入
        if py_file.name == "__init__.py":
            continue

        # 3. 把文件路径 转换为 【相对导入模块名】（核心逻辑）
        # 示例：sub_dir/module_b.py → sub_dir.module_b
        relative_path = py_file.relative_to(package_dir)
        module_name = relative_path.with_suffix("").as_posix().replace("/", ".")

        try:
            # 4. 【相对路径动态导入】（关键：package参数指定当前包）
            module = importlib.import_module(
                name=f".{module_name}",  # 相对导入：加 . 前缀
                package=package_name
            )
            imported_modules.append(module)

        except Exception as e:
            print(f"❌ 导入失败 {module_name}：{str(e)}")




import re

def get_name_by_regex(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'PROVIDER_NAME\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def load_plugin_package(package_path: str):
    init_file = pathlib.Path(package_path).joinpath("__init__.py")
    if not init_file.exists():
        return None
    
    provider_name = get_name_by_regex(init_file)
    if not provider_name:
        return None
    provider_name = f"xpu_perf_provider_{provider_name}"

    spec = importlib.util.spec_from_file_location(provider_name, init_file,
        submodule_search_locations=[str(pathlib.Path(package_path))])
    module = importlib.util.module_from_spec(spec)
    sys.modules[provider_name] = module
    spec.loader.exec_module(module)
    parse_vendor_ops(provider_name, init_file)


def discover_plugins(root_path: str) -> Dict[str, any]:
    visited = set()
    for dirpath, _, filenames in os.walk(root_path):
        if any(dirpath.startswith(p) for p in visited):
            continue
        if "__init__.py" in filenames:
            load_plugin_package(dirpath)
            visited.add(dirpath)




def parse_tasks(task_dir, task):
    """
    默认使用task_dir下递归遍历得到的 **task.json** 文件。
    """
    task_dict : Dict[str, List[Dict[str, Any]]] = {}

    if task_dir is not None:
        task_dir = pathlib.Path(task_dir).absolute()
        if not task_dir.exists():
            logger.error(f"Task dir {task_dir} not exists")
            sys.exit(1)

        json_file_list = list(task_dir.rglob("*.json"))
        
        all_test_cases = {}
        for json_file in json_file_list:
            cur_task_cases = parse_json_file(json_file)

            for kernel, test_cases in cur_task_cases.items():
                if kernel not in all_test_cases:
                    all_test_cases[kernel] = []
                all_test_cases[kernel].extend(test_cases)


        target_op_set = set()
        if task == "all":
            task_dict = all_test_cases
        else:
            for required_task in task.split(","):
                required_task = required_task.strip()
                target_op_set.add(required_task)
            task_dict = {k: v for k, v in all_test_cases.items() if k in target_op_set}
    return task_dict


def parse_workload(workload):
    """
    解析指定的 json or csv 文件
    其中 json文件是列表, 通过笛卡尔积的方式生成所有参数组合, 方便快速生成所有测试用例。
    csv文件是逗号分隔的文件, 第一行是表头, 后面是所有参数组合。
    """
    task_dict : Dict[str, List[Dict[str, Any]]] = {}
    if workload is not None:
        workload_path = pathlib.Path(workload).absolute()
        if not workload_path.exists():
            logger.error(f"Workload file {workload_path} not exists")
            sys.exit(1)

        if workload_path.suffix == ".json":
            task_dict.update(parse_json_file(workload_path))
        elif workload_path.suffix == ".csv":
            task_dict.update(parse_csv_file(workload_path))
        else:
            logger.error(f"Workload file {workload_path} not support, only support json or csv format")
            sys.exit(1)

    return task_dict


def parse_replay_tasks(replay_dir):
    """
    解析replay_dir下的所有 ** rank_*.json **文件, 对齐batch解析所有rank的test cases
    """
    replay_dir = pathlib.Path(replay_dir).absolute()
    if not replay_dir.exists() or not replay_dir.is_dir():
        return {}
    
    replay_task_dict: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for rank_file in replay_dir.glob("rank_*.json"):
        rank_id = int(rank_file.stem.split("_")[-1])
        replay_task_dict[rank_id] = parse_workload(rank_file)

    sorted_items_by_key = sorted(replay_task_dict.items(), key=lambda x: x[0])
    replay_task_dict = dict(sorted_items_by_key)

    return replay_task_dict


def export_reports(
    given_report_dir, 
    info_dict, 
    test_cases={}, 
    bench_results={}
):
    print("*"*100)
    print(f"reports")
    print("*"*100)
    report_dir = pathlib.Path(given_report_dir).absolute()
    report_dir.mkdir(parents=True, exist_ok=True)

    backend_type = info_dict["backend_type"]
    device_name = info_dict["backend"]["device_name"]
    
    target_info_file = report_dir.joinpath(backend_type, device_name, "info.json")
    target_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_info_file, "w") as f:
        export_dict = copy.deepcopy(info_dict)
        export_dict["common"].pop("numa_configs")
        export_dict["runtime"]["device_ids"] = str(export_dict["runtime"]["device_ids"])
        export_dict["runtime"]["device_mapping"] = str(export_dict["runtime"]["device_mapping"])
        export_dict["runtime"]["numa_order"] = str(export_dict["runtime"]["numa_order"])
        json.dump(export_dict, f, indent=4)

    for op_name in bench_results:
        provider_results = {}
        for argument_dict, all_target_dict in zip(test_cases[op_name], bench_results[op_name]):
            for provider, target_dict in all_target_dict.items():
                if target_dict:
                    if provider not in provider_results:
                        provider_results[provider] = []
                    template_dict = {
                        "sku_name": info_dict["backend"]["device_name"], 
                        "op_name": op_name, 
                        "provider": provider, 
                        "arguments": argument_dict, 
                        "targets": target_dict
                    }
                    provider_results[provider].append(template_dict)
        
        for op_provider, data_list in provider_results.items():
            target_dir = report_dir.joinpath(
                backend_type, 
                device_name, 
                op_name, 
                op_provider
            )
            target_dir.mkdir(parents=True, exist_ok=True)


            target_jsonl_file = target_dir.joinpath(f"{op_name}-{op_provider}.jsonl")
            with jsonlines.open(target_jsonl_file, "w") as f:
                f.write_all(data_list)

            temp_pt = prettytable.PrettyTable(["attr", "value"])
            temp_pt.align = "l"
            temp_pt.add_row(["op_name", op_name])
            temp_pt.add_row(["op_provider", op_provider])
            
            arguments_set = set()
            target_set = set()
            for data in data_list:
                arguments_set.update(data["arguments"].keys())
                target_set.update(data["targets"].keys())  

                print(temp_pt)
                print(data["arguments"])
                print(json.dumps(data["targets"], indent=4))
                print("")

            
            target_csv_file = target_dir.joinpath(f"{op_name}-{op_provider}.csv")
            with open(target_csv_file, "w") as f:
                first_occur_arguments = data_list[0]["arguments"].keys()
                first_occur_targets = data_list[0]["targets"].keys()

                keys = ["sku_name", "op_name", "provider"]
                keys.extend(first_occur_arguments)
                for key in arguments_set - set(first_occur_arguments):
                    keys.append(key)
                keys.extend(first_occur_targets)
                for key in target_set - set(first_occur_targets):
                    keys.append(key)

                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()

                for item in data_list:
                    data_dict = {}
                    data_dict["sku_name"] = item["sku_name"]
                    data_dict["op_name"] = item["op_name"]
                    data_dict["provider"] = item["provider"]
                    data_dict.update(item["arguments"])
                    data_dict.update(item["targets"])
                    writer.writerow(data_dict)


