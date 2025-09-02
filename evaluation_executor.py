# file: evaluation_executor_batch.py (Version 11 - Repo-centric Execution)
#
# 该脚本是 evaluation_executor.py 的批量处理版本。
#
# 主要更新 (基于新要求):
# 1. [CRITICAL REFACTOR] 执行流程重构为以 Repo 为中心。
#    - 脚本现在先规划好所有任务，然后按 repo 逐一处理。
#    - 对于每个 repo，会一次性处理其下所有 condition 和 mode 的评估任务。
# 2. [RESOURCE MANAGEMENT] 实现按 Repo 管理 Docker 生命周期。
#    - 在开始评估一个 repo 前启动其容器，评估完该 repo 的所有任务后立即关闭，
#      显著降低了资源占用并避免了容器的频繁启停。
# 3. [LOGGING] 实现按 Repo 分割日志文件。
#    - 每个 repo 的详细评估日志会被记录到独立的日志文件中 (e.g., logs/climlab.log)，
#      极大地提升了日志的可追溯性和调试效率。
# 4. [STABLE] 保留了之前版本的所有关键修复和健壮性改进。
#
# 运行方式:
# python evaluation_executor_batch.py --experiment_dir experiments_output_...

import argparse
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

# 核心修复所需的库
import ast
try:
    import astunparse
except ImportError:
    print("错误: 'astunparse' 库未安装。请运行 'pip install astunparse'。")
    exit(1)

# --- 全局配置 ---
logger = logging.getLogger()

# 宿主机上的基础目录路径
HOST_PYTHON_REPOS_DIR = Path("python_repos")
HOST_REWRITES_DIR = Path("output_rewrites")
BASE_DATA_DIR = Path("data_collection_align")

# Docker 容器内的路径
CONTAINER_REPO_BASE_PATH = "/app/repo_to_process"

# Pytest 相关配置
PYTEST_TIMEOUT_SECONDS = 60

# --- pass@k 计算辅助函数 (无变动) ---
def combinations(n, k):
    if k < 0 or k > n: return 0
    try: return math.comb(n, k)
    except (AttributeError, ValueError):
        if k < 0 or k > n: return 0;
        if k == 0 or k == n: return 1;
        if k > n // 2: k = n - k;
        res = 1;
        for i in range(k): res = res * (n - i) // (i + 1)
        return res

def calculate_pass_at_k(n, c, k):
    if n - c < k: return 1.0
    denominator = combinations(n, k)
    if denominator == 0: return 0.0
    return 1.0 - (combinations(n - c, k) / denominator)

# --- AST 节点替换器 (无变动) ---
class FunctionInjector(ast.NodeTransformer):
    def __init__(self, target_path: List[str], replacement_node: ast.FunctionDef):
        self.target_path = target_path; self.replacement_node = replacement_node; self.current_path = []; self.node_replaced = False
    def visit_ClassDef(self, node: ast.ClassDef):
        self.current_path.append(node.name)
        if len(self.target_path) > 1 and self.target_path[:-1] == self.current_path: self.generic_visit(node)
        self.current_path.pop(); return node
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        path_to_check = self.current_path + [node.name]
        if path_to_check == self.target_path:
            self.node_replaced = True; return self.replacement_node
        return self.generic_visit(node)


class EvaluationExecutor:
    # [MODIFIED] __init__ 现在直接接收数据，不再需要 data_dir
    def __init__(self, predictions_data: Dict, ground_truth_data: Dict, condition: str, mode: str, output_dir: Path, k_values: list):
        self.predictions_data = predictions_data
        self.ground_truth_data = ground_truth_data
        self.condition = condition
        self.mode = mode
        self.output_dir = output_dir
        self.k_values = k_values
        self.answer_pattern = re.compile(r"```output\s*\n(.*?)\s*```", re.DOTALL)
        self.output_file = output_dir / f"{self.mode}.jsonl"
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        # 以追加模式写入，因为一个文件可能由多个 repo 的结果构成
        self.pass_at_k_totals = defaultdict(float)
        self.total_problems = len(self.predictions_data)

    def _generate_modified_code_ast(self, full_file_content: str, target_path: List[str], injected_function_code: str) -> str:
        try:
            tree = ast.parse(full_file_content)
            injected_module = ast.parse(injected_function_code)
            if not (injected_module.body and isinstance(injected_module.body[0], ast.FunctionDef)):
                logging.error(f"注入的代码片段不是一个有效的函数定义。"); return None
            replacement_node = injected_module.body[0]
            injector = FunctionInjector(target_path, replacement_node)
            modified_tree = injector.visit(tree)
            if not injector.node_replaced:
                 logging.error(f"AST 遍历完成，但在文件中未找到路径为 '{'::'.join(target_path)}' 的函数/方法。"); return None
            return astunparse.unparse(modified_tree)
        except Exception as e:
            logging.error(f"使用 AST 修改代码失败 (路径: '{'::'.join(target_path)}'): {e}", exc_info=True); return None

    def _get_test_file_content(self, gt_item: Dict[str, Any]) -> str:
        reponame, testpath = gt_item['reponame'], gt_item['testpath']
        if self.condition == 'original':
            source_file_path = HOST_PYTHON_REPOS_DIR / reponame / testpath
        else:
            path = Path(testpath)
            rewrite_filename = f"{path.stem}_agent_rewrite.py"
            # [BUGFIX] 直接在 reponame 目录下查找 rewrite 文件，移除了错误的 path.parent
            source_file_path = HOST_REWRITES_DIR / reponame / rewrite_filename
        if not source_file_path.exists():
            raise FileNotFoundError(f"测试源文件未找到: {source_file_path}")
        return source_file_path.read_text(encoding='utf-8')

    def _prepare_and_run_in_container(self, gt_item: Dict[str, Any], prediction_text: str) -> bool:
        reponame = gt_item['reponame']; container_name = f"agent-run-{reponame}"
        target_path = [gt_item['classname'], gt_item['funcname']] if gt_item.get('classname') else [gt_item['funcname']]
        pytest_selector = "::".join(target_path)
        original_test_path = Path(gt_item['testpath'])
        temp_filename = f"{original_test_path.stem}_eval_tmp.py"
        temp_test_relative_path = original_test_path.parent / temp_filename
        container_temp_test_full_path = Path(CONTAINER_REPO_BASE_PATH) / temp_test_relative_path
        try:
            full_file_content = self._get_test_file_content(gt_item)
            masked_code = gt_item['masked_code']
            injected_function_code = masked_code.replace("'???'", prediction_text)
            modified_file_content = self._generate_modified_code_ast(full_file_content, target_path, injected_function_code)
            if modified_file_content is None:
                logging.error(f"  任务 {gt_item['task_id']} 的代码注入失败，计为执行失败。"); return False
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py", encoding='utf-8') as tmp_file:
                tmp_file.write(modified_file_content); local_tmp_path = tmp_file.name
            subprocess.run(["docker", "cp", local_tmp_path, f"{container_name}:{container_temp_test_full_path.as_posix()}"], check=True, capture_output=True)
            os.unlink(local_tmp_path)
            relative_posix_path = temp_test_relative_path.as_posix()
            pytest_command = f"pytest --timeout={PYTEST_TIMEOUT_SECONDS} {relative_posix_path}::{pytest_selector}"
            exec_result = subprocess.run(["docker", "exec", "--workdir", CONTAINER_REPO_BASE_PATH, container_name, "bash", "-c", pytest_command], capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if exec_result.returncode == 0:
                logging.info(f"  [成功] 任务 {gt_item['task_id']} 测试通过。"); return True
            else:
                failure_reason = "未知原因"
                if "FAILED" in exec_result.stdout: failure_reason = "测试断言失败 (FAILED)"
                elif "ERROR" in exec_result.stdout: failure_reason = "测试执行出错 (ERROR)"
                elif "Timeout >" in exec_result.stdout or "Timeout >" in exec_result.stderr: failure_reason = "测试执行超时 (Timeout)"
                logging.warning(f"  [失败] 任务 {gt_item['task_id']} 测试失败 ({failure_reason})，返回码: {exec_result.returncode}。")
                logging.debug(f"  [Pytest STDOUT]:\n{exec_result.stdout}"); logging.debug(f"  [Pytest STDERR]:\n{exec_result.stderr}"); return False
        except Exception as e:
            logging.error(f"发生意外错误 (任务 {gt_item['task_id']}): {e}", exc_info=True); return False
        finally:
            check_running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name], capture_output=True, text=True)
            if check_running.returncode == 0 and check_running.stdout.strip() == 'true':
                 path_to_remove = container_temp_test_full_path.as_posix()
                 subprocess.run(["docker", "exec", container_name, "rm", "-f", path_to_remove], capture_output=True)

    def run_evaluation(self):
        if self.total_problems == 0: return {}
        # [MODIFIED] 不再打开文件，而是返回结果给主循环
        all_results = {}
        for i, (task_id, predictions) in enumerate(self.predictions_data.items()):
            logging.info(f"--- 正在处理任务 {i+1}/{self.total_problems}: {task_id} ---")
            gt_item = self.ground_truth_data[task_id]
            num_correct = 0; sample_results, extracted_predictions = [], []
            for pred in predictions:
                response = pred.get('response', ''); match = self.answer_pattern.search(response)
                prediction_text = match.group(1).strip() if match else ""
                extracted_predictions.append(prediction_text)
                if not match or not prediction_text:
                    logging.warning(f"  任务 {task_id} 的一个预测未能提取到有效代码或代码为空，计为失败。"); sample_results.append(False); continue
                is_correct = self._prepare_and_run_in_container(gt_item, prediction_text)
                if is_correct: num_correct += 1
                sample_results.append(is_correct)
            num_samples = len(predictions)
            task_pass_k = {k: calculate_pass_at_k(num_samples, num_correct, k) for k in self.k_values}
            result_log = {"task_id": task_id, "reponame": gt_item['reponame'], "condition": self.condition, "mode": self.mode, "ground_truth": gt_item['ground_truth'], "predictions": extracted_predictions, "execution_results": sample_results, "num_correct": num_correct, "num_samples": num_samples, "pass_at_k": task_pass_k}
            
            # [MODIFIED] 将结果写入文件，并累加 pass@k
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_log) + '\n')
            for k in self.k_values:
                all_results[k] = all_results.get(k, 0.0) + task_pass_k[k]

        return all_results

# [NEW] 日志设置函数，用于按 repo 分割日志
def setup_repo_logger(log_dir: Path, reponame: str):
    """配置 logger，为指定的 repo 创建一个文件处理器。"""
    log_file = log_dir / f"{reponame}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 移除旧的文件处理器，避免日志重复写入
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    # 添加新的文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(reponame)s] - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # 使用 Filter 为日志记录添加 reponame 上下文
    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.reponame = reponame
            return True
    logger.addFilter(ContextFilter())
    logger.addHandler(file_handler)

def main():
    parser = argparse.ArgumentParser(description="批量运行所有实验组合的基于执行的评估。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--experiment_dir", type=str, required=True, help="顶层实验输出目录, 例如:\n'experiments_output_deepseek_deepseek-reasoner_10240'")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="设置日志输出级别。")
    args = parser.parse_args()

    # --- 基本路径和日志配置 ---
    base_experiment_dir = Path(args.experiment_dir)
    if not base_experiment_dir.is_dir(): print(f"错误: 实验目录 '{base_experiment_dir}' 不存在。"); return
    exp_name = base_experiment_dir.name
    dynamic_output_dir_name = exp_name.replace("experiments_output_", "evaluation_results_execution_")
    EVALUATION_OUTPUT_DIR = Path(dynamic_output_dir_name)
    EVALUATION_OUTPUT_DIR.mkdir(exist_ok=True)
    LOGS_DIR = EVALUATION_OUTPUT_DIR / "logs" # [NEW] 日志子目录
    LOGS_DIR.mkdir(exist_ok=True)
    
    # 配置根 logger (只配置控制台输出)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    if logger.hasHandlers(): logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(stream_handler)

    logging.info(f"所有评估结果将被保存到: '{EVALUATION_OUTPUT_DIR}'")
    logging.info(f"详细的 repo 日志将被保存到: '{LOGS_DIR}'")

    # --- [REFACTOR] 阶段一：规划所有评估任务 ---
    logging.info("===== 阶段一：正在规划所有评估任务... =====")
    
    # 1. 加载所有基准数据
    ground_truth_data = {}
    for file_path in BASE_DATA_DIR.glob("**/*.jsonl"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line); ground_truth_data[item['task_id']] = item
    
    # 2. 构建任务计划，按 repo -> condition -> mode 组织
    evaluation_plan = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    CONDITIONS = ["original", "rewrite"]
    k_values = [1, 2, 3, 4, 5]

    for condition in CONDITIONS:
        predictions_base_dir = base_experiment_dir / condition / "predictions"
        if not predictions_base_dir.is_dir(): continue
        for mode_dir in predictions_base_dir.iterdir():
            if not mode_dir.is_dir(): continue
            mode = mode_dir.name
            for pred_file in mode_dir.glob("*.jsonl"):
                with open(pred_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        pred = json.loads(line)
                        task_id = pred['task_id']
                        if task_id in ground_truth_data:
                            reponame = ground_truth_data[task_id]['reponame']
                            evaluation_plan[reponame][condition][mode][task_id].append(pred)
    
    logging.info(f"任务规划完成，共发现 {len(evaluation_plan)} 个仓库需要评估。")

    # --- [REFACTOR] 阶段二：按 Repo 顺序执行评估 ---
    logging.info("===== 阶段二：开始按仓库顺序执行评估 =====")
    
    pass_at_k_reports = defaultdict(lambda: defaultdict(lambda: {'total_pass_k': defaultdict(float), 'total_problems': 0}))

    for reponame in sorted(evaluation_plan.keys()):
        print("\n" + "#"*80 + f"\n#  开始处理仓库: {reponame.upper()}\n" + "#"*80)
        setup_repo_logger(LOGS_DIR, reponame)
        container_name = f"agent-run-{reponame}"

        try:
            # 1. 启动并准备容器
            logging.info(f"正在为仓库 '{reponame}' 启动并准备容器 '{container_name}'...")
            subprocess.run(["docker", "start", container_name], check=True, capture_output=True)
            dep_install_cmd = "pip install -q pytest-timeout"
            subprocess.run(["docker", "exec", container_name, "bash", "-c", dep_install_cmd], check=True, capture_output=True)
            logging.info(f"容器 '{container_name}' 已就绪。")
            
            # 2. 执行该 repo 下的所有评估
            for condition in evaluation_plan[reponame]:
                for mode in evaluation_plan[reponame][condition]:
                    logging.info(f"--- 开始评估组合: {condition.upper()} / {mode.upper()} for repo {reponame} ---")
                    predictions_to_run = evaluation_plan[reponame][condition][mode]
                    output_dir = EVALUATION_OUTPUT_DIR / condition
                    
                    evaluator = EvaluationExecutor(predictions_to_run, ground_truth_data, condition, mode, output_dir, k_values)
                    repo_pass_k_results = evaluator.run_evaluation()
                    
                    # 累加报告结果
                    report = pass_at_k_reports[condition][mode]
                    report['total_problems'] += len(predictions_to_run)
                    for k, val in repo_pass_k_results.items():
                        report['total_pass_k'][k] += val

        except subprocess.CalledProcessError as e:
            logging.error(f"为仓库 '{reponame}' 准备或执行 Docker 命令时失败！跳过该仓库。")
            logging.error(f"  Stderr: {e.stderr.decode('utf-8', 'ignore') if e.stderr else 'N/A'}")
        except Exception as e:
            logging.error(f"处理仓库 '{reponame}' 时发生未知错误: {e}", exc_info=True)
        finally:
            # 3. 停止容器
            logging.info(f"仓库 '{reponame}' 处理完毕，正在停止容器 '{container_name}'...")
            try:
                subprocess.run(["docker", "stop", container_name], capture_output=True, check=False) # check=False to avoid error if already stopped
            except Exception as e:
                logging.warning(f"停止容器 '{container_name}' 时出错: {e}")

    # --- 阶段三：打印最终报告 ---
    logging.info("\n" + "="*60 + "\n最终评估报告汇总\n" + "="*60)
    for condition in sorted(pass_at_k_reports.keys()):
        for mode in sorted(pass_at_k_reports[condition].keys()):
            report = pass_at_k_reports[condition][mode]
            total_problems = report['total_problems']
            print("\n" + "-"*60 + f"\n报告: {condition.upper()} / {mode.upper()}\n" + f"总计评估了 {total_problems} 个独立任务。\n" + "-"*60)
            if total_problems == 0:
                print("没有结果可报告。")
                continue
            print("\n+--------+----------+\n|   k    |  pass@k  |\n+--------+----------+")
            for k in sorted(k_values):
                avg_pass_k = report['total_pass_k'][k] / total_problems if total_problems > 0 else 0.0
                print(f"| pass@{k:<2} | {avg_pass_k:<8.4f} |")
            print("+--------+----------+\n")

    logging.info("\n===== 所有批量评估任务已完成 =====")

if __name__ == "__main__":
    main()