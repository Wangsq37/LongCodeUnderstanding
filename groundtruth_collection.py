# file: ground_truth_collector.py (FINAL, v5, Pytest -s Fix)
#
# 该脚本用于通过在 Docker 容器中执行代码来获取 ground truth 表达式的实际运行时值。
#
# 核心功能:
# 1. [CRITICAL FIX 5] 在 pytest 命令中添加 '-s' 标志，以禁用输出捕获。
#    这确保了我们注入的 print 语句能够被子进程捕获，解决了“测试通过但结果丢失”的根本问题。
# 2. 回退到更简单、更可靠的“打印标记”方案，移除了不必要的文件 I/O 复杂性。
# 3. 使用 AST (抽象语法树) 进行精确的代码注入。
# 4. 高效的批量处理和结构化结果存储。

import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple, List

import ast
try:
    import astunparse
except ImportError:
    print("错误: 'astunparse' 库未安装。请运行 'pip install astunparse'。")
    exit(1)

# --- 全局配置 ---
logger = logging.getLogger(__name__)
HOST_PYTHON_REPOS_DIR = Path("python_repos")
HOST_REWRITES_DIR = Path("output_rewrites")
BASE_DATA_DIR = Path("data_collection_align")
CONTAINER_REPO_BASE_PATH = "/app/repo_to_process"
PYTEST_TIMEOUT_SECONDS = 120

# --- AST 注入器 (回退到 v3 的打印版本) ---
class AssertReplacer(ast.NodeTransformer):
    def __init__(self, target_path: List[str], ground_truth_expr: str, injection_code: str):
        self.target_path = target_path; self.ground_truth_expr = ground_truth_expr
        self.replacement_nodes = ast.parse(injection_code).body
        self.current_path = []; self.node_replaced = False; self.in_target_function = False
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.current_path.append(node.name)
        if len(self.target_path) > 1 and self.target_path[:-1] == self.current_path: self.generic_visit(node)
        self.current_path.pop(); return node
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        path_to_check = self.current_path + [node.name]
        if path_to_check == self.target_path:
            self.in_target_function = True; self.generic_visit(node); self.in_target_function = False
        return node
    def visit_Assert(self, node: ast.Assert) -> Any:
        if not self.in_target_function or self.node_replaced: return node
        if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            right_side_expr = astunparse.unparse(node.test.comparators[0]).strip()
            if right_side_expr == self.ground_truth_expr:
                self.node_replaced = True; return self.replacement_nodes
        return node

# --- 核心逻辑函数 (已更新) ---
def get_test_file_content(gt_item: Dict[str, Any]) -> str:
    reponame, testpath = gt_item['reponame'], gt_item['testpath']
    condition = gt_item['condition']
    if condition == 'original':
        source_file_path = HOST_PYTHON_REPOS_DIR / reponame / testpath
    else:
        path = Path(testpath)
        rewrite_filename = f"{path.stem}_agent_rewrite.py"
        source_file_path = HOST_REWRITES_DIR / reponame / rewrite_filename
    if not source_file_path.exists():
        raise FileNotFoundError(f"测试源文件未找到: {source_file_path}")
    return source_file_path.read_text(encoding='utf-8')

def generate_modified_content_ast(original_content: str, gt_item: Dict[str, Any]) -> str:
    ground_truth_expr = gt_item['ground_truth']
    target_path = [gt_item['classname'], gt_item['funcname']] if gt_item.get('classname') else [gt_item['funcname']]
    injection_code = f"""
# --- Injected code by ground_truth_collector ---
try:
    import json, sys
    _gt_value = ({ground_truth_expr})
    _serializable_value = {{"status": "success", "value": _gt_value, "type": str(type(_gt_value))}}
    try: output = json.dumps(_serializable_value)
    except (TypeError, OverflowError):
        _serializable_value['value'] = repr(_gt_value)
        output = json.dumps(_serializable_value)
    print("\\n---GT_START---"); print(output); print("---GT_END---")
    sys.stdout.flush() # 强制刷新缓冲区
except Exception as e:
    import traceback
    error_info = {{"status": "failure", "error": str(e), "traceback": traceback.format_exc()}}
    print("\\n---GT_START---"); print(json.dumps(error_info)); print("---GT_END---")
    sys.stdout.flush() # 强制刷新缓冲区
# --- End of injected code ---
"""
    try:
        tree = ast.parse(original_content)
        replacer = AssertReplacer(target_path, ground_truth_expr, injection_code.strip())
        modified_tree = replacer.visit(tree)
        if not replacer.node_replaced:
            raise ValueError(f"AST 遍历完成，但在函数 '{'::'.join(target_path)}' 中未找到匹配的 assert 语句。")
        ast.fix_missing_locations(modified_tree)
        return astunparse.unparse(modified_tree)
    except Exception as e:
        logger.error(f"使用 AST 修改代码失败 (Task ID: {gt_item['task_id']}): {e}")
        raise

def get_runtime_value_in_container(gt_item: Dict[str, Any], container_name: str) -> Dict:
    """在容器中执行修改后的代码并通过 stdout 获取运行时值。"""
    try:
        original_content = get_test_file_content(gt_item)
        modified_content = generate_modified_content_ast(original_content, gt_item)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py", encoding='utf-8') as tmp_file:
            tmp_file.write(modified_content)
            local_tmp_path = tmp_file.name

        testpath_in_repo = Path(gt_item['testpath'])
        temp_filename = f"{testpath_in_repo.stem}_gt_collector_tmp.py"
        temp_test_relative_path = testpath_in_repo.parent / temp_filename
        container_temp_test_full_path = Path(CONTAINER_REPO_BASE_PATH) / temp_test_relative_path
        
        container_path_str = container_temp_test_full_path.as_posix()
        subprocess.run(["docker", "cp", local_tmp_path, f"{container_name}:{container_path_str}"], check=True, capture_output=True)
        os.unlink(local_tmp_path)
        
        pytest_selector = "::".join([gt_item['classname'], gt_item['funcname']] if gt_item.get('classname') else [gt_item['funcname']])
        
        # --- FIX: 添加 -s 标志以禁用输出捕获 ---
        pytest_command = f"pytest -s --timeout={PYTEST_TIMEOUT_SECONDS} {container_path_str}::{pytest_selector}"
        
        exec_result = subprocess.run(
            ["docker", "exec", "--workdir", CONTAINER_REPO_BASE_PATH, container_name, "bash", "-c", pytest_command],
            capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        stdout = exec_result.stdout
        if "---GT_START---" in stdout and "---GT_END---" in stdout:
            result_str = stdout.split("---GT_START---")[1].split("---GT_END---")[0].strip()
            return json.loads(result_str)
        else:
            # 如果测试提前失败，我们仍然可以从 stdout 中获得一些线索
            return {"status": "failure", "error": "Execution markers not found in output.", "details": f"Pytest returncode: {exec_result.returncode}\nPytest stdout:\n{stdout}\nPytest stderr:\n{exec_result.stderr}"}
    
    except Exception as e:
        logger.error(f"处理任务 {gt_item['task_id']} 时发生意外错误: {e}", exc_info=False)
        return {"status": "failure", "error": f"Unexpected exception: {str(e)}"}
    finally:
        if 'container_temp_test_full_path' in locals():
            subprocess.run(["docker", "exec", container_name, "rm", "-f", container_temp_test_full_path.as_posix()], capture_output=True)

# --- main 函数及之后的部分保持不变 ---
def main():
    parser = argparse.ArgumentParser(description="通过执行获取 Ground Truth 的运行时值。")
    parser.add_argument("--output_dir", type=str, default="groundtruth_collection", help="存放输出结果的目录路径。")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="设置日志输出级别。")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"所有输出将被保存到目录: '{OUTPUT_DIR}'")

    original_output_file = OUTPUT_DIR / "original.jsonl"
    rewrite_output_file = OUTPUT_DIR / "rewrite.jsonl"

    if original_output_file.exists(): original_output_file.unlink()
    if rewrite_output_file.exists(): rewrite_output_file.unlink()

    logging.info("===== 阶段一：正在规划所有任务... =====")
    evaluation_plan = defaultdict(list)
    CONDITIONS = ["original", "rewrite"]
    for condition in CONDITIONS:
        data_dir_for_condition = BASE_DATA_DIR / condition
        if not data_dir_for_condition.exists():
            logging.warning(f"数据目录 '{data_dir_for_condition}' 不存在，跳过。")
            continue
        for file_path in data_dir_for_condition.glob("**/*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    item['condition'] = condition
                    reponame = item['reponame']
                    evaluation_plan[reponame].append(item)
    
    total_repos = len(evaluation_plan)
    total_tasks = sum(len(tasks) for tasks in evaluation_plan.values())
    logging.info(f"任务规划完成，共发现 {total_repos} 个仓库，{total_tasks} 个任务需要处理。")

    logging.info("===== 阶段二：开始按仓库顺序执行 =====")
    processed_tasks = 0
    file_handles = {}
    try:
        file_handles['original'] = open(original_output_file, 'a', encoding='utf-8')
        file_handles['rewrite'] = open(rewrite_output_file, 'a', encoding='utf-8')

        for i, (reponame, tasks) in enumerate(sorted(evaluation_plan.items())):
            print("\n" + "#"*80 + f"\n#  ({i+1}/{total_repos}) 开始处理仓库: {reponame.upper()}\n" + "#"*80)
            container_name = f"agent-run-{reponame}"
            try:
                logging.info(f"正在为仓库 '{reponame}' 启动容器 '{container_name}'...")
                subprocess.run(["docker", "start", container_name], check=True, capture_output=True)
                for task in tasks:
                    task_id = task['task_id']
                    condition = task['condition']
                    logging.info(f"--- 正在处理任务: {task_id} (Condition: {condition}) ---")
                    result_data = get_runtime_value_in_container(task, container_name)
                    final_log = {"task_id": task_id, "reponame": reponame, "condition": condition, "original_ground_truth": task['ground_truth'], "runtime_ground_truth": result_data}
                    output_file_handle = file_handles[condition]
                    output_file_handle.write(json.dumps(final_log) + '\n')
                    processed_tasks += 1
                    logging.info(f"--- 任务 {task_id} 完成，状态: {result_data.get('status', 'unknown')} ---")
            except subprocess.CalledProcessError as e:
                logging.error(f"为仓库 '{reponame}' 准备 Docker 命令时失败！跳过该仓库。")
                logging.error(f"  Stderr: {e.stderr.decode('utf-8', 'ignore') if e.stderr else 'N/A'}")
            except Exception as e:
                logging.error(f"处理仓库 '{reponame}' 时发生未知错误: {e}", exc_info=True)
            finally:
                logging.info(f"仓库 '{reponame}' 处理完毕，正在停止容器 '{container_name}'...")
                subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
    finally:
        for handle in file_handles.values():
            if handle: handle.close()
    
    logging.info("\n" + "="*60 + "\n所有任务处理完毕！\n" + "="*60)
    logging.info(f"总计处理了 {processed_tasks}/{total_tasks} 个任务。")
    logging.info(f"运行时 Ground Truth 值已保存到目录: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()