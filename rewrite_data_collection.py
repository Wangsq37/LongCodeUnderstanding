# import ast
# import os
# import json
# import astunparse
# import copy
# import sys
# from collections import defaultdict
# from typing import Union, Dict, List

# # 导入数据质量过滤器
# try:
#     from data_quality_filter import create_quality_filter
# except ImportError:
#     print("警告: 数据质量过滤器模块未找到，将使用基础过滤。", file=sys.stderr)
#     create_quality_filter = None

# # ==============================================================================
# # 阶段 1: 数据提取的核心代码
# # (从 data_collection_rewrite.py 继承)
# # ==============================================================================

# class TaskIdGenerator:
#     """一个简单的线程不安全的计数器，用于生成唯一的task ID。"""
#     def __init__(self):
#         self.current_id = 0
    
#     def next(self) -> int:
#         task_id = self.current_id
#         self.current_id += 1
#         return task_id

# class AssertionTransformer(ast.NodeTransformer):
#     """AST转换器，用于遮蔽断言语句的右侧。"""
#     def __init__(self, target_lineno: int, target_col_offset: int):
#         self.target_lineno = target_lineno
#         self.target_col_offset = target_col_offset
#         self.transformed = False
#         self.ground_truth = None

#     def visit_Assert(self, node: ast.Assert) -> ast.Assert:
#         if not (node.lineno == self.target_lineno and node.col_offset == self.target_col_offset):
#             return node
#         if not (isinstance(node.test, ast.Compare) and
#                 node.test.ops and isinstance(node.test.ops[0], ast.Eq)):
#             return node
#         original_node = node.test.comparators[0]
#         self.ground_truth = astunparse.unparse(original_node).strip()
#         self.transformed = True
#         try:
#             node.test.comparators[0] = ast.Constant(value='???', kind=None)
#         except AttributeError:
#             node.test.comparators[0] = ast.Name(id='???', ctx=ast.Load())
#         return node

# def process_rewritten_test_file(file_path: str, repo_root: str, task_id_gen: TaskIdGenerator):
#     """解析单个重写后的测试文件，并提取数据。现在集成了数据质量过滤功能。"""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             source_code = f.read()
#         tree = ast.parse(source_code)
#     except (SyntaxError, UnicodeDecodeError, PermissionError) as e:
#         print(f"警告: 无法解析 {file_path}。原因: {e}。已跳过。", file=sys.stderr)
#         return []

#     imports_list = [astunparse.unparse(node).strip() for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
#     all_data_entries = []
#     reponame = os.path.basename(repo_root)
#     original_testname = os.path.basename(file_path)
#     normalized_testname = original_testname.replace('_agent_rewrite.py', '.py') if original_testname.endswith('_agent_rewrite.py') else original_testname
#     original_relpath = os.path.relpath(file_path, repo_root)
#     normalized_relpath = original_relpath.replace(original_testname, normalized_testname)
    
#     # 初始化质量过滤器
#     quality_filter = create_quality_filter() if create_quality_filter else None

#     for node in ast.walk(tree):
#         if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
#             # 使用质量过滤器筛选断言
#             if quality_filter:
#                 quality_asserts = quality_filter.filter_test_function(node)
#                 candidate_asserts = [item["assert_node"] for item in quality_asserts]
#             else:
#                 # 回退到基础过滤
#                 candidate_asserts = [sn for sn in ast.walk(node) if isinstance(sn, ast.Assert) and isinstance(sn.test, ast.Compare) and sn.test.ops and isinstance(sn.test.ops[0], ast.Eq)]
            
#             for assert_to_mask in candidate_asserts:
#                 func_copy = copy.deepcopy(node)
#                 transformer = AssertionTransformer(assert_to_mask.lineno, assert_to_mask.col_offset)
#                 masked_function_node = transformer.visit(func_copy)
#                 if transformer.transformed and transformer.ground_truth is not None:
#                     task_id_str = f"{reponame}_{task_id_gen.next()}"
                    
#                     # 获取质量分析信息
#                     quality_analysis = None
#                     if quality_filter:
#                         analysis = quality_filter.analyze_assertion_complexity(assert_to_mask)
#                         quality_analysis = analysis
                    
#                     data_entry = {
#                         "task_id": task_id_str, "reponame": reponame,
#                         "testpath": normalized_relpath.replace('\\', '/'),
#                         "testname": normalized_testname, "funcname": node.name,
#                         "imports": imports_list, "code": astunparse.unparse(node).strip(),
#                         "masked_code": astunparse.unparse(masked_function_node).strip(),
#                         "ground_truth": transformer.ground_truth
#                     }
                    
#                     # 添加质量分析信息
#                     if quality_analysis:
#                         data_entry["quality_analysis"] = quality_analysis
                    
#                     all_data_entries.append(data_entry)
#     return all_data_entries

# def run_data_collection_stage(repos_dir: str, output_dir: str):
#     """阶段1的执行函数：遍历仓库并从重写文件中提取数据。"""
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"提取的数据将临时保存到: {os.path.abspath(output_dir)}")
#     repo_names = [d for d in os.listdir(repos_dir) if os.path.isdir(os.path.join(repos_dir, d))]
#     if not repo_names:
#         print(f"错误: 在 '{repos_dir}' 目录中未找到任何仓库子目录。", file=sys.stderr)
#         return False

#     for reponame in repo_names:
#         repo_path = os.path.join(repos_dir, reponame)
#         output_file = os.path.join(output_dir, f"{reponame}.jsonl")
#         task_id_gen = TaskIdGenerator()
#         print(f"\n--- 正在处理仓库: {reponame} ---")
#         all_processed_data = []
#         for root, _, files in os.walk(repo_path):
#             for file in files:
#                 if file.endswith("_agent_rewrite.py"):
#                     file_path = os.path.join(root, file)
#                     processed_data = process_rewritten_test_file(file_path, repo_path, task_id_gen)
#                     all_processed_data.extend(processed_data)
#         if not all_processed_data:
#             print(f"在 {reponame} 中未找到符合条件的断言。")
#             continue
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for entry in all_processed_data:
#                 f.write(json.dumps(entry, ensure_ascii=False) + '\n')
#         print(f"✅ 成功! 找到 {len(all_processed_data)} 个断言。数据已保存至: {output_file}")
#     return True

# # ==============================================================================
# # 阶段 2: 难度分析的核心代码
# # (从 analysis.py 继承)
# # ==============================================================================

# def normalize_func_name(name: str) -> str:
#     """将 report_functions.jsonl 中的复杂函数名规范化。"""
#     return name.split('[')[0].split('::')[-1]

# def get_difficulty(dependencies: list) -> str:
#     """根据依赖列表计算难度等级。"""
#     if not dependencies: return 'easy'
#     max_hops = max(dep.get('hops', 0) for dep in dependencies)
#     if max_hops <= 2: return 'easy'
#     return 'hard'

# def load_difficulty_map(report_file_path: str) -> Union[Dict[str, str], None]:
#     """加载 report_functions.jsonl 文件并创建难度映射。"""
#     if not os.path.exists(report_file_path): return None
#     difficulty_map = {}
#     with open(report_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 norm_name = normalize_func_name(data['test_function'])
#                 if norm_name not in difficulty_map:
#                     difficulty_map[norm_name] = get_difficulty(data.get('dependencies', []))
#             except (json.JSONDecodeError, KeyError): continue
#     return difficulty_map

# def run_analysis_stage(collection_dir: str, reports_dir: str, final_output_dir: str):
#     """阶段2的执行函数：过滤并分析已提取的数据。"""
#     os.makedirs(final_output_dir, exist_ok=True)
#     print(f"过滤后的最终数据将被保存到: {os.path.abspath(final_output_dir)}")
#     total_stats = {'lines_kept': 0, 'lines_discarded': 0, 'func_counts': defaultdict(int), 'total_unique_funcs': 0}
#     data_files = [f for f in os.listdir(collection_dir) if f.endswith('.jsonl')]
#     if not data_files:
#         print(f"警告: 在临时目录 '{collection_dir}' 中未找到任何 .jsonl 文件可供分析。", file=sys.stderr)
#         return

#     for filename in data_files:
#         reponame = filename.replace('.jsonl', '')
#         print(f"\n--- 正在分析仓库: {reponame} ---")
#         report_file = os.path.join(reports_dir, reponame, 'report_functions.jsonl')
#         difficulty_map = load_difficulty_map(report_file)
#         if difficulty_map is None:
#             print(f"  警告: 未找到报告文件 '{report_file}'。跳过此仓库的分析。\n")
#             continue

#         input_path = os.path.join(collection_dir, filename)
#         output_path = os.path.join(final_output_dir, filename)
#         lines_kept, lines_discarded = 0, 0
#         repo_func_stats = defaultdict(int)
#         counted_funcs_in_repo = set()

#         with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
#             for line in infile:
#                 try:
#                     assert_data = json.loads(line)
#                     funcname = assert_data.get('funcname')
#                     if funcname and funcname in difficulty_map:
#                         outfile.write(line)
#                         lines_kept += 1
#                         if funcname not in counted_funcs_in_repo:
#                             difficulty = difficulty_map[funcname]
#                             repo_func_stats[difficulty] += 1
#                             counted_funcs_in_repo.add(funcname)
#                     else:
#                         lines_discarded += 1
#                 except (json.JSONDecodeError, KeyError):
#                     lines_discarded += 1
        
#         total_lines = lines_kept + lines_discarded
#         print(f"  [过滤统计] 共处理 {total_lines} 行数据:")
#         if total_lines > 0:
#             print(f"    - 保留: {lines_kept} 行 ({lines_kept/total_lines:.1%})")
#             print(f"    - 丢弃: {lines_discarded} 行 ({lines_discarded/total_lines:.1%})")

#         num_unique_funcs = len(counted_funcs_in_repo)
#         print(f"  [难度统计] 在保留的数据中分析了 {num_unique_funcs} 个唯一的测试函数:")
#         if num_unique_funcs > 0:
#             for difficulty in ['easy', 'hard']:
#                 count = repo_func_stats[difficulty]
#                 print(f"    - {difficulty.capitalize():<7}: {count} ({count/num_unique_funcs:.1%})")

#         total_stats['lines_kept'] += lines_kept
#         total_stats['lines_discarded'] += lines_discarded
#         total_stats['total_unique_funcs'] += num_unique_funcs
#         for key, value in repo_func_stats.items():
#             total_stats['func_counts'][key] += value

#     print("\n" + "=" * 60)
#     print("--- 总体分析报告 ---")
#     grand_total_lines = total_stats['lines_kept'] + total_stats['lines_discarded']
#     print(f"\n[总体过滤统计] 所有仓库共处理 {grand_total_lines} 行数据:")
#     if grand_total_lines > 0:
#         print(f"  - 总计保留: {total_stats['lines_kept']} 行 ({total_stats['lines_kept']/grand_total_lines:.1%})")
#         print(f"  - 总计丢弃: {total_stats['lines_discarded']} 行 ({total_stats['lines_discarded']/grand_total_lines:.1%})")

#     total_funcs = total_stats['total_unique_funcs']
#     print(f"\n[总体难度统计] 所有保留的数据共包含 {total_funcs} 个唯一的测试函数:")
#     if total_funcs > 0:
#         for difficulty in ['easy', 'hard']:
#             count = total_stats['func_counts'][difficulty]
#             print(f"  - {difficulty.capitalize():<7}: {count} ({count/total_funcs:.1%})")
#     print("=" * 60)

# # ==============================================================================
# # 主逻辑: 协调两个阶段
# # ==============================================================================

# def main():
#     """
#     主函数，按顺序执行数据提取和难度分析两个阶段。
#     """
#     current_dir = os.getcwd()
#     # 定义输入目录
#     rewrites_dir = os.path.join(current_dir, 'output_rewrites')
#     reports_dir = os.path.join(current_dir, 'output_results')
    
#     # 定义中间和最终输出目录
#     collection_temp_dir = os.path.join(current_dir, 'data_collection_from_rewrites')
#     final_filtered_dir = os.path.join(current_dir, 'data_collection_from_rewrites_filtered')

#     # --- 检查输入目录是否存在 ---
#     if not os.path.isdir(rewrites_dir) or not os.path.isdir(reports_dir):
#         print(f"错误: 请确保 'output_rewrites' 和 'output_results' 目录都存在于当前工作目录。", file=sys.stderr)
#         return

#     # --- 阶段 1: 从重写文件中提取数据 ---
#     print("="*25 + " 阶段 1: 提取数据 " + "="*25)
#     success = run_data_collection_stage(rewrites_dir, collection_temp_dir)
#     if not success:
#         print("阶段 1 未能成功执行。流水线中止。", file=sys.stderr)
#         return

#     # --- 阶段 2: 对提取的数据进行难度分类和过滤 ---
#     print("\n" + "="*20 + " 阶段 2: 分析、过滤与分类 " + "="*20)
#     run_analysis_stage(collection_temp_dir, reports_dir, final_filtered_dir)
    
#     print("\n--- 所有任务已完成 ---")

# if __name__ == '__main__':
#     try:
#         import astunparse
#     except ImportError:
#         print("错误: 'astunparse' 库未安装。", file=sys.stderr)
#         print("请使用以下命令进行安装: pip install astunparse", file=sys.stderr)
#         sys.exit(1)
        
#     main()
import ast
import os
import json
import astunparse
import copy
import sys
from collections import defaultdict
from typing import Union, Dict, List

# 导入数据质量过滤器
try:
    from data_quality_filter import create_quality_filter
except ImportError:
    print("警告: 数据质量过滤器模块未找到，将使用基础过滤。", file=sys.stderr)
    create_quality_filter = None

# ==============================================================================
# [NEW] AST 访问器，用于查找测试并记录其类上下文 (与 data_collection.py 相同)
# ==============================================================================
class TestFinder(ast.NodeVisitor):
    def __init__(self):
        self.found_tests = []
        self._current_class_name = None

    def visit_ClassDef(self, node: ast.ClassDef):
        original_class_name = self._current_class_name
        self._current_class_name = node.name
        self.generic_visit(node)
        self._current_class_name = original_class_name

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name.startswith("test_"):
            self.found_tests.append({
                "node": node,
                "classname": self._current_class_name
            })
        self.generic_visit(node)
        
# ==============================================================================
# 阶段 1: 数据提取的核心代码 (已修改)
# ==============================================================================
class TaskIdGenerator:
    def __init__(self):
        self.current_id = 0
    def next(self) -> int:
        task_id = self.current_id; self.current_id += 1; return task_id

class AssertionTransformer(ast.NodeTransformer):
    def __init__(self, target_lineno: int, target_col_offset: int):
        self.target_lineno, self.target_col_offset = target_lineno, target_col_offset
        self.transformed, self.ground_truth = False, None
    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        if not (node.lineno == self.target_lineno and node.col_offset == self.target_col_offset and
                isinstance(node.test, ast.Compare) and node.test.ops and isinstance(node.test.ops[0], ast.Eq)):
            return node
        original_node = node.test.comparators[0]
        self.ground_truth = astunparse.unparse(original_node).strip()
        self.transformed = True
        try: node.test.comparators[0] = ast.Constant(value='???', kind=None)
        except AttributeError: node.test.comparators[0] = ast.Name(id='???', ctx=ast.Load())
        return node

def process_rewritten_test_file(file_path: str, repo_root: str, task_id_gen: TaskIdGenerator):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: source_code = f.read()
        tree = ast.parse(source_code)
    except (SyntaxError, UnicodeDecodeError, PermissionError) as e:
        print(f"警告: 无法解析 {file_path}。原因: {e}。已跳过。", file=sys.stderr); return []

    imports_list = [astunparse.unparse(node).strip() for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    all_data_entries, reponame = [], os.path.basename(repo_root)
    original_testname = os.path.basename(file_path)
    normalized_testname = original_testname.replace('_agent_rewrite.py', '.py')
    original_relpath = os.path.relpath(file_path, repo_root)
    normalized_relpath = original_relpath.replace(original_testname, normalized_testname)
    
    quality_filter = create_quality_filter() if create_quality_filter else None

    # [MODIFIED] 使用 TestFinder
    finder = TestFinder()
    finder.visit(tree)

    for test_info in finder.found_tests:
        node, classname = test_info["node"], test_info["classname"]
        
        candidate_asserts = [item["assert_node"] for item in quality_filter.filter_test_function(node)] if quality_filter else [sn for sn in ast.walk(node) if isinstance(sn, ast.Assert) and isinstance(sn.test, ast.Compare) and sn.test.ops and isinstance(sn.test.ops[0], ast.Eq)]
        
        for assert_to_mask in candidate_asserts:
            func_copy = copy.deepcopy(node)
            transformer = AssertionTransformer(assert_to_mask.lineno, assert_to_mask.col_offset)
            masked_function_node = transformer.visit(func_copy)
            if transformer.transformed and transformer.ground_truth is not None:
                task_id_str = f"{reponame}_{task_id_gen.next()}"
                quality_analysis = quality_filter.analyze_assertion_complexity(assert_to_mask) if quality_filter else None
                
                data_entry = {
                    "task_id": task_id_str, "reponame": reponame,
                    "testpath": normalized_relpath.replace('\\', '/'),
                    "testname": normalized_testname,
                    "classname": classname,  # [NEW] 添加 classname 字段
                    "funcname": node.name,
                    "imports": imports_list, "code": astunparse.unparse(node).strip(),
                    "masked_code": astunparse.unparse(masked_function_node).strip(),
                    "ground_truth": transformer.ground_truth
                }
                if quality_analysis: data_entry["quality_analysis"] = quality_analysis
                all_data_entries.append(data_entry)
    return all_data_entries

# ... run_data_collection_stage 和后续函数与 data_collection.py 类似，保持不变 ...
def run_data_collection_stage(repos_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"提取的数据将临时保存到: {os.path.abspath(output_dir)}")
    repo_names = [d for d in os.listdir(repos_dir) if os.path.isdir(os.path.join(repos_dir, d))]
    if not repo_names:
        print(f"错误: 在 '{repos_dir}' 目录中未找到任何仓库子目录。", file=sys.stderr); return False
    for reponame in repo_names:
        repo_path = os.path.join(repos_dir, reponame)
        output_file = os.path.join(output_dir, f"{reponame}.jsonl")
        task_id_gen = TaskIdGenerator()
        print(f"\n--- 正在处理仓库: {reponame} ---")
        all_processed_data = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith("_agent_rewrite.py"):
                    file_path = os.path.join(root, file)
                    processed_data = process_rewritten_test_file(file_path, repo_path, task_id_gen)
                    all_processed_data.extend(processed_data)
        if not all_processed_data:
            print(f"在 {reponame} 中未找到符合条件的断言。"); continue
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_processed_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ 成功! 找到 {len(all_processed_data)} 个断言。数据已保存至: {output_file}")
    return True

# ... 其他函数 (run_analysis_stage, main 等) 保持不变 ...
def normalize_func_name(name: str) -> str:
    return name.split('[')[0].split('::')[-1]
def get_difficulty(dependencies: list) -> str:
    if not dependencies: return 'easy'
    max_hops = max(dep.get('hops', 0) for dep in dependencies)
    return 'easy' if max_hops <= 2 else 'hard'
def load_difficulty_map(report_file_path: str) -> Union[Dict[str, str], None]:
    if not os.path.exists(report_file_path): return None
    difficulty_map = {}
    with open(report_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                norm_name = normalize_func_name(data['test_function'])
                if norm_name not in difficulty_map:
                    difficulty_map[norm_name] = get_difficulty(data.get('dependencies', []))
            except (json.JSONDecodeError, KeyError): continue
    return difficulty_map
def run_analysis_stage(collection_dir: str, reports_dir: str, final_output_dir: str):
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"过滤后的最终数据将被保存到: {os.path.abspath(final_output_dir)}")
    total_stats = {'lines_kept': 0, 'lines_discarded': 0, 'func_counts': defaultdict(int), 'total_unique_funcs': 0}
    data_files = [f for f in os.listdir(collection_dir) if f.endswith('.jsonl')]
    if not data_files:
        print(f"警告: 在临时目录 '{collection_dir}' 中未找到任何 .jsonl 文件可供分析。", file=sys.stderr); return
    for filename in data_files:
        reponame = filename.replace('.jsonl', '')
        print(f"\n--- 正在分析仓库: {reponame} ---")
        report_file = os.path.join(reports_dir, reponame, 'report_functions.jsonl')
        difficulty_map = load_difficulty_map(report_file)
        if difficulty_map is None:
            print(f"  警告: 未找到报告文件 '{report_file}'。跳过此仓库的分析。\n"); continue
        input_path, output_path = os.path.join(collection_dir, filename), os.path.join(final_output_dir, filename)
        lines_kept, lines_discarded = 0, 0
        repo_func_stats, counted_funcs_in_repo = defaultdict(int), set()
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    assert_data = json.loads(line)
                    funcname = assert_data.get('funcname')
                    if funcname and funcname in difficulty_map:
                        outfile.write(line); lines_kept += 1
                        if funcname not in counted_funcs_in_repo:
                            difficulty = difficulty_map[funcname]
                            repo_func_stats[difficulty] += 1
                            counted_funcs_in_repo.add(funcname)
                    else: lines_discarded += 1
                except (json.JSONDecodeError, KeyError): lines_discarded += 1
        total_lines = lines_kept + lines_discarded
        print(f"  [过滤统计] 共处理 {total_lines} 行数据:")
        if total_lines > 0:
            print(f"    - 保留: {lines_kept} 行 ({lines_kept/total_lines:.1%})")
            print(f"    - 丢弃: {lines_discarded} 行 ({lines_discarded/total_lines:.1%})")
        num_unique_funcs = len(counted_funcs_in_repo)
        print(f"  [难度统计] 在保留的数据中分析了 {num_unique_funcs} 个唯一的测试函数:")
        if num_unique_funcs > 0:
            for difficulty in ['easy', 'hard']:
                count = repo_func_stats[difficulty]
                print(f"    - {difficulty.capitalize():<7}: {count} ({count/num_unique_funcs:.1%})")
        total_stats['lines_kept'] += lines_kept; total_stats['lines_discarded'] += lines_discarded
        total_stats['total_unique_funcs'] += num_unique_funcs
        for key, value in repo_func_stats.items(): total_stats['func_counts'][key] += value
    print("\n" + "=" * 60 + "\n--- 总体分析报告 ---")
    grand_total_lines = total_stats['lines_kept'] + total_stats['lines_discarded']
    print(f"\n[总体过滤统计] 所有仓库共处理 {grand_total_lines} 行数据:")
    if grand_total_lines > 0:
        print(f"  - 总计保留: {total_stats['lines_kept']} 行 ({total_stats['lines_kept']/grand_total_lines:.1%})")
        print(f"  - 总计丢弃: {total_stats['lines_discarded']} 行 ({total_stats['lines_discarded']/grand_total_lines:.1%})")
    total_funcs = total_stats['total_unique_funcs']
    print(f"\n[总体难度统计] 所有保留的数据共包含 {total_funcs} 个唯一的测试函数:")
    if total_funcs > 0:
        for difficulty in ['easy', 'hard']:
            count = total_stats['func_counts'][difficulty]
            print(f"  - {difficulty.capitalize():<7}: {count} ({count/total_funcs:.1%})")
    print("=" * 60)

def main():
    current_dir = os.getcwd()
    rewrites_dir = os.path.join(current_dir, 'output_rewrites')
    reports_dir = os.path.join(current_dir, 'output_results')
    collection_temp_dir = os.path.join(current_dir, 'data_collection_from_rewrites')
    final_filtered_dir = os.path.join(current_dir, 'data_collection_from_rewrites_filtered')
    if not os.path.isdir(rewrites_dir) or not os.path.isdir(reports_dir):
        print(f"错误: 请确保 'output_rewrites' 和 'output_results' 目录都存在于当前工作目录。", file=sys.stderr); return
    print("="*25 + " 阶段 1: 提取数据 " + "="*25)
    success = run_data_collection_stage(rewrites_dir, collection_temp_dir)
    if not success:
        print("阶段 1 未能成功执行。流水线中止。", file=sys.stderr); return
    print("\n" + "="*20 + " 阶段 2: 分析、过滤与分类 " + "="*20)
    run_analysis_stage(collection_temp_dir, reports_dir, final_filtered_dir)
    print("\n--- 所有任务已完成 ---")

if __name__ == '__main__':
    try: import astunparse
    except ImportError:
        print("错误: 'astunparse' 库未安装。\n请使用以下命令进行安装: pip install astunparse", file=sys.stderr); sys.exit(1)
    main()