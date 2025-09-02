# file: augment_groundtruth.py (FIXED v2)
#
# 该脚本用于合并原始数据集 (data_collection_align) 和
# 运行时获取的 ground truth 值 (groundtruth_collection)，
# 生成一个增强版的新数据集 (data_collection_align_groundtruth)，
# 其中 'ground_truth' 字段是一个包含多个可能值的列表。
#
# [FIX v2]: 增强了对 '_pytest.python_api.ApproxScalar' 类型的处理。
# - 如果 `pytest.approx` 的参数是表达式 (如 `test[1]`)，脚本现在会
#   从运行时的 `value` 字符串 (如 "-45.6 ± ...") 中正确提取数值。

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Set

def process_runtime_value(runtime_info: Dict[str, Any], original_gt: str) -> List[str]:
    """
    根据运行时信息的类型和值，以及定义的规则，提取新的 ground truth 字符串。
    """
    if not runtime_info or runtime_info.get("status") != "success":
        return []

    value = runtime_info.get("value")
    type_str = runtime_info.get("type", "")
    new_truths: Set[str] = set()

    if type_str == "<class 'NoneType'>":
        return []
    
    # [MODIFIED] 增强对 ApproxScalar 的处理
    elif type_str == "<class '_pytest.python_api.ApproxScalar'>":
        # 策略 1: 优先尝试从原始 GT 字符串中解析出数字字面量
        match = re.search(r"pytest\.approx\(([^,)]+)", original_gt)
        if match:
            num_str = match.group(1).strip()
            try:
                # 如果成功，说明参数是 "1", "0.5" 这样的字面量
                num = float(num_str)
                if num.is_integer():
                    new_truths.add(str(int(num)))
                    new_truths.add(str(float(num)))
                else:
                    new_truths.add(str(num))
                # 成功解析字面量后，直接返回
                return list(new_truths)
            except ValueError:
                # 如果解析失败 (例如，参数是 'test[1]'), 则忽略错误，
                # 并自动进入下面的回退策略。
                pass

        # 策略 2 (回退): 解析运行时的 value 字符串 (e.g., "-45.6 \u00b1 4.6e-05")
        if isinstance(value, str) and '\u00b1' in value:
            # 提取 '±' 符号之前的部分
            num_part_str = value.split('\u00b1')[0].strip()
            try:
                # 将提取出的部分转换为浮点数并添加
                float_val = float(num_part_str)
                new_truths.add(str(float_val))
            except ValueError:
                # 如果解析失败，则忽略
                pass
        
        return list(new_truths)

    elif type_str in ["<class 'numpy.int32'>", "<class 'numpy.int64'>", "<class 'numpy.uint8'>"]:
        new_truths.add(str(value))
        return list(new_truths)
    elif type_str == "<class 'tuple'>":
        if isinstance(value, list):
            new_truths.add(str(tuple(value)))
        return list(new_truths)
    elif type_str == "<class 'str'>":
        new_truths.add(repr(value))
        return list(new_truths)
    else:
        new_truths.add(str(value))
        return list(new_truths)

def main():
    parser = argparse.ArgumentParser(description="增强 Ground Truth 数据集。")
    parser.add_argument("--data_dir", type=str, default="data_collection_align", help="原始数据集的目录路径。")
    parser.add_argument("--runtime_dir", type=str, default="groundtruth_collection", help="运行时 Ground Truth 数据的目录路径。")
    parser.add_argument("--output_dir", type=str, default="data_collection_align_groundtruth", help="存放增强后数据集的目录路径。")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    RUNTIME_DIR = Path(args.runtime_dir)
    OUTPUT_DIR = Path(args.output_dir)

    if not DATA_DIR.is_dir() or not RUNTIME_DIR.is_dir():
        print(f"错误: 请确保输入目录 '{DATA_DIR}' 和 '{RUNTIME_DIR}' 存在。")
        return

    conditions = ["original", "rewrite"]
    for condition in conditions:
        print("\n" + "#"*60)
        print(f"# 正在处理 '{condition.upper()}' 数据")
        print("#"*60)

        runtime_file = RUNTIME_DIR / f"{condition}.jsonl"
        source_data_dir = DATA_DIR / condition
        output_data_dir = OUTPUT_DIR / condition

        if not runtime_file.exists():
            print(f"警告: 未找到运行时文件 '{runtime_file}'，跳过 '{condition}'。")
            continue
        if not source_data_dir.is_dir():
            print(f"警告: 未找到原始数据目录 '{source_data_dir}'，跳过 '{condition}'。")
            continue

        print(f"正在从 '{runtime_file}' 加载运行时数据...")
        runtime_map: Dict[str, Dict] = {}
        with open(runtime_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                runtime_map[data['task_id']] = data
        print(f"加载完成，共找到 {len(runtime_map)} 条 '{condition}' 运行时记录。")

        source_files = list(source_data_dir.glob("**/*.jsonl"))
        if not source_files:
            print(f"警告: 在 '{source_data_dir}' 中未找到任何 .jsonl 文件。")
            continue

        output_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"正在处理 '{source_data_dir}' 中的文件，并写入到 '{output_data_dir}'...")

        for source_file_path in source_files:
            relative_path = source_file_path.relative_to(source_data_dir)
            output_file_path = output_data_dir / relative_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  -> 正在处理 {source_file_path} -> {output_file_path}")

            with open(source_file_path, 'r', encoding='utf-8') as f_in, \
                 open(output_file_path, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    task_data = json.loads(line)
                    task_id = task_data['task_id']
                    original_gt = task_data['ground_truth']
                    
                    all_gts: Set[str] = {original_gt}
                    
                    if task_id in runtime_map:
                        runtime_info = runtime_map[task_id].get("runtime_ground_truth", {})
                        new_truths = process_runtime_value(runtime_info, original_gt)
                        
                        for truth in new_truths:
                            all_gts.add(truth)
                    
                    task_data['ground_truth'] = sorted(list(all_gts))
                    f_out.write(json.dumps(task_data) + '\n')

    print("\n所有处理已完成！")
    print(f"增强后的数据集已保存在: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()