# file: enhanced_evaluation.py
#
# 这是对原始评估脚本的修改版，以支持增强后的 ground truth 数据。
#
# 核心更新:
# 1. [DATA SOURCE] 从 'data_collection_align_groundtruth' 加载数据，
#    其中的 'ground_truth' 字段是一个列表。
# 2. [EVALUATION LOGIC] 核心评估逻辑被修改为：只要模型的预测匹配
#    ground truth 列表中的任意一个值，即判为正确。
# 3. [ROBUST COMPARISON] 引入了基于 `ast.literal_eval()` 的智能比较函数，
#    以解决因引号 (' vs ") 或其他格式问题导致的误判。

import os
import json
import re
import math
import argparse
import ast  # 引入 ast 库用于智能比较
from collections import defaultdict
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    PREDICTIONS_DIR_BASE = None
    EVALUATION_RESULTS_DIR = None
    
    # [MODIFIED] 数据源目录已更新
    DATA_DIR_BASE = Path("data_collection_align_groundtruth")
    
    DIFFICULTY_REPORTS_DIR = Path("output_results")
    ANSWER_PATTERN = re.compile(r"```output\s*\n(.*?)\s*```", re.DOTALL)
    K_VALUES = [1, 2, 3, 4, 5]
    NUM_SAMPLES_PER_TASK = 5

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_jsonl(file_path):
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: print(f"⚠️ WARNING: Skipping malformed JSON line in {file_path}")
    return data

def normalize_funcname(name: str) -> str:
    name = re.sub(r'\[.*\]$', '', name)
    if '::' in name: name = name.split('::')[-1]
    return name

def combinations(n, k):
    if k < 0 or k > n: return 0
    return math.comb(n, k)

def calculate_pass_at_k(n, c, k):
    if n - c < k: return 1.0
    denominator = combinations(n, k)
    if denominator == 0: return 0.0
    return 1.0 - (combinations(n - c, k) / denominator)

# [NEW] 智能比较函数
def robust_compare(prediction_str: str, ground_truth_str: str) -> bool:
    """
    使用 ast.literal_eval 进行智能比较，以处理格式差异（如引号）。
    如果解析失败，则回退到字符串比较。
    """
    prediction_str = prediction_str.strip()
    ground_truth_str = ground_truth_str.strip()
    
    try:
        # 尝试将两者都解析为 Python 字面量
        pred_val = ast.literal_eval(prediction_str)
        gt_val = ast.literal_eval(ground_truth_str)
        
        # 比较解析后的对象
        return pred_val == gt_val
    except (ValueError, SyntaxError, MemoryError, TypeError):
        # 如果任一解析失败（例如，它们是复杂表达式），则回退到简单的字符串比较
        return prediction_str == ground_truth_str

# [MODIFIED] 核心评估函数，现在接收一个 GT 列表
def is_prediction_correct(prediction: str, ground_truth_list: list) -> bool:
    """
    检查预测是否与 ground truth 列表中的任意一个值匹配。
    """
    for gt_val in ground_truth_list:
        if robust_compare(prediction, str(gt_val)):
            return True
    return False

def print_pass_k_table(title, pass_k_data, num_problems):
    print(f"\n📊 Pass@k Results ({title}) - {num_problems} problems")
    if num_problems == 0: print("   No data available for calculation."); return
    final_pass_k = {k: v / num_problems for k, v in pass_k_data.items()}
    print("+--------+----------+")
    print("|   k    |  pass@k  |")
    print("+--------+----------+")
    for k in Config.K_VALUES:
        val = final_pass_k.get(k, 0.0)
        print(f"| pass@{k:<2} | {val:<8.4f} |")
    print("+--------+----------+")

# ==============================================================================
# MAIN ORCHESTRATOR (Minor modifications for new data structure)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate code completion experiments and generate detailed reports.")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Base directory of the experiment predictions to evaluate.")
    args = parser.parse_args()

    Config.PREDICTIONS_DIR_BASE = Path(args.predictions_dir)
    if not Config.PREDICTIONS_DIR_BASE.exists():
        print(f"FATAL: Predictions directory not found at '{Config.PREDICTIONS_DIR_BASE}'")
        exit(1)
        
    base_name = Config.PREDICTIONS_DIR_BASE.name
    eval_name = base_name.replace("experiments_output_", "evaluation_results_")
    Config.EVALUATION_RESULTS_DIR = Path(eval_name)
    
    print("===== Starting Enhanced Experiment Evaluation (using augmented ground truths) =====")
    print(f"  - Input Predictions: {Config.PREDICTIONS_DIR_BASE}")
    print(f"  - Ground Truths from: {Config.DATA_DIR_BASE}")
    print(f"  - Output Reports:    {Config.EVALUATION_RESULTS_DIR}")
    Config.EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ["retrieval", "confusion", "oracle"]:
        print(f"\n\n{'='*30} Evaluating Mode: {mode.upper()} {'='*30}")
        
        if not (Config.PREDICTIONS_DIR_BASE / "original" / "predictions" / mode).exists():
            print(f"Prediction directory for mode '{mode}' not found. Skipping.")
            continue

        output_files = {}
        for cond in ["original", "rewrite"]:
            output_dir = Config.EVALUATION_RESULTS_DIR / cond / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_dir / "evaluation.jsonl"
            output_files[cond] = open(output_file_path, 'w', encoding='utf-8')
            print(f"  - Detailed results for '{cond}' will be saved to: {output_file_path}")

        print("\n--- PASS 1: Loading predictions, ground truths, and difficulty reports...")
        
        all_ground_truths = {"original": {}, "rewrite": {}}
        all_predictions = {"original": defaultdict(list), "rewrite": defaultdict(list)}
        task_to_difficulty_map = {}

        repo_list = sorted([f.stem for f in (Config.PREDICTIONS_DIR_BASE / "original" / "predictions" / mode).glob('*.jsonl')])
        
        for reponame in repo_list:
            for cond in ["original", "rewrite"]:
                # [MODIFIED] 从新目录加载数据
                gt_file = Config.DATA_DIR_BASE / cond / f"{reponame}.jsonl"
                for item in load_jsonl(gt_file):
                    all_ground_truths[cond][item['task_id']] = item

            for cond in ["original", "rewrite"]:
                pred_file = Config.PREDICTIONS_DIR_BASE / cond / "predictions" / mode / f"{reponame}.jsonl"
                for pred in load_jsonl(pred_file):
                    all_predictions[cond][pred['task_id']].append(pred)
            
            difficulty_report_file = Config.DIFFICULTY_REPORTS_DIR / reponame / "report_functions.jsonl"
            funcname_to_difficulty = {}
            for report in load_jsonl(difficulty_report_file):
                normalized_name = normalize_funcname(report['test_function'])
                max_hops = max(dep.get('hops', 0) for dep in report.get('dependencies', [])) if report.get('dependencies') else 0
                difficulty = 'easy' if max_hops <= 2 else 'hard'
                funcname_to_difficulty[normalized_name] = difficulty
            
            # 使用 original 条件的数据来构建 difficulty map (因为 funcname 不变)
            for task_id, item in all_ground_truths['original'].items():
                if item['reponame'] == reponame:
                    task_to_difficulty_map[task_id] = funcname_to_difficulty.get(item['funcname'], 'unknown')

        print("\n--- PASS 2: Aligning datasets based on available predictions...")
        original_task_ids_with_preds = set(all_predictions["original"].keys())
        rewrite_task_ids_with_preds = set(all_predictions["rewrite"].keys())
        common_task_ids = original_task_ids_with_preds.intersection(rewrite_task_ids_with_preds)
        
        print(f"Tasks with predictions in 'original': {len(original_task_ids_with_preds)}")
        print(f"Tasks with predictions in 'rewrite':  {len(rewrite_task_ids_with_preds)}")
        print(f"Common tasks for evaluation:      {len(common_task_ids)}")
        
        if not common_task_ids:
            print("No common tasks found. Skipping evaluation for this mode.")
            continue

        print("\n--- PASS 3: Calculating Pass@k and saving detailed results...")
        final_summary = defaultdict(lambda: defaultdict(lambda: {"pass_k": defaultdict(float), "num_problems": 0}))

        for task_id in sorted(list(common_task_ids)):
            difficulty = task_to_difficulty_map.get(task_id, 'unknown')
            
            for cond in ["original", "rewrite"]:
                gt_item = all_ground_truths[cond][task_id]
                samples = all_predictions[cond][task_id]
                
                # [MODIFIED] 使用新的评估函数
                num_correct = 0
                extracted_predictions = []
                for sample in samples:
                    response = sample.get('response', '')
                    match = Config.ANSWER_PATTERN.search(response)
                    extracted_pred = match.group(1).strip() if match else ""
                    extracted_predictions.append(extracted_pred)
                    
                    # 核心改动：用 is_prediction_correct 替代旧的比较
                    if match and is_prediction_correct(extracted_pred, gt_item['ground_truth']):
                        num_correct += 1
                
                n, c = Config.NUM_SAMPLES_PER_TASK, num_correct
                
                task_pass_k = {k: calculate_pass_at_k(n, c, k) for k in Config.K_VALUES}
                eval_result = {
                    "task_id": task_id,
                    "reponame": gt_item['reponame'],
                    "condition": cond,
                    "mode": mode,
                    "difficulty": difficulty,
                    "ground_truth": gt_item['ground_truth'], # 记录 GT 列表
                    "predictions": extracted_predictions,
                    "num_correct": c,
                    "num_samples": n,
                    "pass_at_k": task_pass_k
                }
                output_files[cond].write(json.dumps(eval_result, ensure_ascii=False) + '\n')

                for group in [difficulty, 'overall']:
                    summary_group = final_summary[cond][group]
                    summary_group['num_problems'] += 1
                    for k, pass_k_value in task_pass_k.items():
                        summary_group['pass_k'][k] += pass_k_value

        for cond in ["original", "rewrite"]:
            print(f"\n\n{'~'*20} Summary for: {cond.upper()} / {mode.upper()} {'~'*20}")
            summary_data = final_summary[cond]
            for difficulty_level in ['easy', 'hard', 'unknown', 'overall']:
                if difficulty_level in summary_data:
                    data = summary_data[difficulty_level]
                    print_pass_k_table(difficulty_level, data['pass_k'], data['num_problems'])

        for f in output_files.values():
            f.close()

    print("\n\n===== Evaluation Script Finished =====")


if __name__ == "__main__":
    main()

# import os
# import json
# import re
# import argparse
# from collections import defaultdict
# from pathlib import Path

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================
# class Config:
#     # These will be set by command-line arguments
#     PREDICTIONS_DIR_BASE = None
#     EVALUATION_RESULTS_DIR = None
    
#     # Static paths
#     DATA_DIR_BASE = Path("data_collection_align")
#     DIFFICULTY_REPORTS_DIR = Path("output_results")
    
#     # Constants
#     ANSWER_PATTERN = re.compile(r"```output\s*\n(.*?)\s*```", re.DOTALL)
#     K_VALUES = [1, 2, 3, 4, 5]
#     # NOTE: Updated to reflect that only one sample is generated per task.
#     NUM_SAMPLES_PER_TASK = 1

# # ==============================================================================
# # HELPER FUNCTIONS
# # ==============================================================================
# def load_jsonl(file_path):
#     """Loads a JSONL file into a list of dictionaries."""
#     data = []
#     if not file_path.exists(): return data
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try: 
#                 data.append(json.loads(line))
#             except json.JSONDecodeError: 
#                 print(f"⚠️ WARNING: Skipping malformed JSON line in {file_path}")
#     return data

# def normalize_funcname(name: str) -> str:
#     """Normalizes a function name by removing templates and namespaces."""
#     name = re.sub(r'\[.*\]$', '', name)
#     if '::' in name:
#         name = name.split('::')[-1]
#     return name

# def compare_values(response_val, ground_truth_val):
#     """Compares response and ground truth values, attempting numeric comparison first."""
#     response_val = str(response_val).strip()
#     ground_truth_val = str(ground_truth_val).strip()
#     try: 
#         return float(response_val) == float(ground_truth_val)
#     except (ValueError, TypeError): 
#         return response_val == ground_truth_val

# def print_pass_k_table(title, pass_k_data, num_problems):
#     """Prints a formatted table of pass@k results."""
#     print(f"\n📊 Pass@k Results ({title}) - {num_problems} problems")
#     if num_problems == 0: 
#         print("   No data available for calculation.")
#         return
        
#     # NOTE: For n=1, all pass@k values will be identical.
#     final_pass_k = {k: v / num_problems for k, v in pass_k_data.items()}
#     print("+--------+----------+")
#     print("|   k    |  pass@k  |")
#     print("+--------+----------+")
#     for k in Config.K_VALUES:
#         val = final_pass_k.get(k, 0.0)
#         print(f"| pass@{k:<2} | {val:<8.4f} |")
#     print("+--------+----------+")

# # ==============================================================================
# # MAIN ORCHESTRATOR
# # ==============================================================================
# def main():
#     parser = argparse.ArgumentParser(description="Evaluate code completion experiments and generate detailed reports.")
#     parser.add_argument("--predictions_dir", type=str, required=True, help="Base directory of the experiment predictions to evaluate.")
#     args = parser.parse_args()

#     # --- Dynamically configure paths based on input ---
#     Config.PREDICTIONS_DIR_BASE = Path(args.predictions_dir)
#     if not Config.PREDICTIONS_DIR_BASE.exists():
#         print(f"FATAL: Predictions directory not found at '{Config.PREDICTIONS_DIR_BASE}'")
#         exit(1)
        
#     base_name = Config.PREDICTIONS_DIR_BASE.name
#     eval_name = base_name.replace("experiments_output_", "evaluation_results_")
#     Config.EVALUATION_RESULTS_DIR = Path(eval_name)
    
#     print("===== Starting Enhanced Experiment Evaluation =====")
#     print(f"  - Input Predictions: {Config.PREDICTIONS_DIR_BASE}")
#     print(f"  - Output Reports:    {Config.EVALUATION_RESULTS_DIR}")
#     Config.EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#     # --- Evaluate all modes, including 'oracle' ---
#     for mode in ["retrieval", "confusion", "oracle"]:
#         print(f"\n\n{'='*30} Evaluating Mode: {mode.upper()} {'='*30}")
        
#         # Check if prediction data exists for this mode
#         if not (Config.PREDICTIONS_DIR_BASE / "original" / "predictions" / mode).exists():
#             print(f"Prediction directory for mode '{mode}' not found. Skipping.")
#             continue

#         # --- Setup detailed output files for this mode ---
#         output_files = {}
#         for cond in ["original", "rewrite"]:
#             output_dir = Config.EVALUATION_RESULTS_DIR / cond / mode
#             output_dir.mkdir(parents=True, exist_ok=True)
#             output_file_path = output_dir / "evaluation.jsonl"
#             output_files[cond] = open(output_file_path, 'w', encoding='utf-8')
#             print(f"  - Detailed results for '{cond}' will be saved to: {output_file_path}")

#         # --- PASS 1: Pre-load all data into memory ---
#         print("\n--- PASS 1: Loading predictions, ground truths, and difficulty reports...")
        
#         all_ground_truths = {"original": {}, "rewrite": {}}
#         all_predictions = {"original": defaultdict(list), "rewrite": defaultdict(list)}
#         task_to_difficulty_map = {}

#         repo_list = sorted([f.stem for f in (Config.PREDICTIONS_DIR_BASE / "original" / "predictions" / mode).glob('*.jsonl')])
        
#         for reponame in repo_list:
#             # Load ground truths
#             for cond in ["original", "rewrite"]:
#                 for item in load_jsonl(Config.DATA_DIR_BASE / cond / f"{reponame}.jsonl"):
#                     all_ground_truths[cond][item['task_id']] = item

#             # Load predictions
#             for cond in ["original", "rewrite"]:
#                 pred_file = Config.PREDICTIONS_DIR_BASE / cond / "predictions" / mode / f"{reponame}.jsonl"
#                 for pred in load_jsonl(pred_file):
#                     all_predictions[cond][pred['task_id']].append(pred)
            
#             # Load difficulty report and build map
#             difficulty_report_file = Config.DIFFICULTY_REPORTS_DIR / reponame / "report_functions.jsonl"
#             funcname_to_difficulty = {}
#             for report in load_jsonl(difficulty_report_file):
#                 normalized_name = normalize_funcname(report['test_function'])
#                 max_hops = max(dep.get('hops', 0) for dep in report.get('dependencies', [])) if report.get('dependencies') else 0
#                 difficulty = 'easy' if max_hops <= 2 else 'hard'
#                 funcname_to_difficulty[normalized_name] = difficulty
            
#             # Map task_ids to their difficulty
#             for task_id, item in all_ground_truths['original'].items():
#                 if item['reponame'] == reponame:
#                     task_to_difficulty_map[task_id] = funcname_to_difficulty.get(item['funcname'], 'unknown')

#         # --- PASS 2: Find common task_ids for fair comparison ---
#         print("\n--- PASS 2: Aligning datasets based on available predictions...")
#         original_task_ids_with_preds = set(all_predictions["original"].keys())
#         rewrite_task_ids_with_preds = set(all_predictions["rewrite"].keys())
#         common_task_ids = original_task_ids_with_preds.intersection(rewrite_task_ids_with_preds)
        
#         print(f"Tasks with predictions in 'original': {len(original_task_ids_with_preds)}")
#         print(f"Tasks with predictions in 'rewrite':  {len(rewrite_task_ids_with_preds)}")
#         print(f"Common tasks for evaluation:      {len(common_task_ids)}")
        
#         if not common_task_ids:
#             print("No common tasks found. Skipping evaluation for this mode.")
#             continue

#         # --- PASS 3: Evaluate, save detailed results, and aggregate for summary ---
#         print("\n--- PASS 3: Calculating Pass@k and saving detailed results...")
#         final_summary = defaultdict(lambda: defaultdict(lambda: {"pass_k": defaultdict(float), "num_problems": 0}))

#         for task_id in sorted(list(common_task_ids)): # Sort for deterministic output
#             difficulty = task_to_difficulty_map.get(task_id, 'unknown')
            
#             for cond in ["original", "rewrite"]:
#                 gt_item = all_ground_truths[cond][task_id]
#                 samples = all_predictions[cond][task_id]
                
#                 num_correct = 0
#                 extracted_predictions = []
#                 for sample in samples:
#                     response = sample.get('response', '')
#                     match = Config.ANSWER_PATTERN.search(response)
#                     extracted_pred = match.group(1).strip() if match else f"ERROR: Pattern not found in '{response[:50]}...'"
#                     extracted_predictions.append(extracted_pred)
#                     if match and compare_values(extracted_pred, gt_item['ground_truth']):
#                         num_correct += 1
                
#                 # --- Simplified Pass@k Calculation for n=1 ---
#                 # If the single sample is correct (c > 0), pass rate is 1.0. Otherwise, it's 0.0.
#                 # This rate is the same for all k values (pass@1, pass@2, etc.).
#                 pass_rate = 1.0 if num_correct > 0 else 0.0
#                 task_pass_k = {k: pass_rate for k in Config.K_VALUES}

#                 # --- Log detailed results to JSONL file ---
#                 eval_result = {
#                     "task_id": task_id,
#                     "reponame": gt_item['reponame'],
#                     "condition": cond,
#                     "mode": mode,
#                     "difficulty": difficulty,
#                     "ground_truth": gt_item['ground_truth'],
#                     "predictions": extracted_predictions,
#                     "num_correct": num_correct,
#                     "num_samples": len(samples),
#                     "pass_at_k": task_pass_k
#                 }
#                 output_files[cond].write(json.dumps(eval_result, ensure_ascii=False) + '\n')

#                 # --- Aggregate results for the final summary table ---
#                 for group in [difficulty, 'overall']:
#                     summary_group = final_summary[cond][group]
#                     summary_group['num_problems'] += 1
#                     for k, pass_k_value in task_pass_k.items():
#                         summary_group['pass_k'][k] += pass_k_value

#         # --- Print Final Summary Tables for this mode ---
#         for cond in ["original", "rewrite"]:
#             print(f"\n\n{'~'*20} Summary for: {cond.upper()} / {mode.upper()} {'~'*20}")
#             summary_data = final_summary[cond]
#             for difficulty_level in ['easy', 'hard', 'unknown', 'overall']:
#                 if difficulty_level in summary_data:
#                     data = summary_data[difficulty_level]
#                     print_pass_k_table(difficulty_level, data['pass_k'], data['num_problems'])

#         # --- Clean up file handlers for this mode ---
#         for f in output_files.values():
#             f.close()

#     print("\n\n===== Evaluation Script Finished =====")


# if __name__ == "__main__":
#     main()