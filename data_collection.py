# # -*- coding: utf-8 -*-

# import ast
# import os
# import json
# import copy
# import sys

# # 尝试导入 astunparse，如果失败则提示用户安装
# try:
#     import astunparse
# except ImportError:
#     print("错误: 'astunparse' 库未安装。", file=sys.stderr)
#     print("请使用以下命令进行安装: pip install astunparse", file=sys.stderr)
#     sys.exit(1)

# # 导入数据质量过滤器
# try:
#     from data_quality_filter import create_quality_filter
# except ImportError:
#     print("警告: 数据质量过滤器模块未找到，将使用基础过滤。", file=sys.stderr)
#     create_quality_filter = None

# # ==============================================================================
# # 核心数据提取代码 (源自旧版本，保持不变)
# # ==============================================================================

# class TaskIdGenerator:
#     """一个简单的线程不安全的计数器，用于生成唯一的task ID。"""
#     def __init__(self):
#         self.current_id = 0
    
#     def next(self) -> int:
#         """返回当前ID并递增计数器。"""
#         task_id = self.current_id
#         self.current_id += 1
#         return task_id

# class AssertionTransformer(ast.NodeTransformer):
#     """
#     一个AST转换器，其功能是：
#     1. 寻找由行号和列号指定的 *单个* `assert` 语句。
#     2. 如果该语句包含等式检查 (`==`)，则将其右侧替换为 '???'。
#     3. 将原始右侧内容记录为 ground truth（字符串形式）。
#     """
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
        
#         # 兼容不同 Python 版本的 AST 节点
#         try:
#             node.test.comparators[0] = ast.Constant(value='???', kind=None)
#         except AttributeError:
#             node.test.comparators[0] = ast.Name(id='???', ctx=ast.Load())
            
#         return node

# def process_test_file(file_path: str, repo_root: str, task_id_gen: TaskIdGenerator):
#     """
#     解析一个 Python 测试文件，找到包含等式断言的测试函数，
#     并为 *每一个* 这样的断言创建一条结构化数据。
#     现在集成了数据质量过滤功能。
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             source_code = f.read()
#         tree = ast.parse(source_code)
#     except (SyntaxError, UnicodeDecodeError, PermissionError, FileNotFoundError) as e:
#         print(f"警告: 无法解析 {file_path}。原因: {e}。已跳过。", file=sys.stderr)
#         return []

#     imports_list = [astunparse.unparse(node).strip() for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]

#     all_data_entries = []
#     reponame = os.path.basename(repo_root)
#     testpath = os.path.relpath(file_path, repo_root)
#     testname = os.path.basename(file_path)
    
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
#                 candidate_asserts = []
#                 for sub_node in ast.walk(node):
#                     if (isinstance(sub_node, ast.Assert) and
#                         isinstance(sub_node.test, ast.Compare) and
#                         sub_node.test.ops and isinstance(sub_node.test.ops[0], ast.Eq)):
#                         candidate_asserts.append(sub_node)
            
#             for assert_to_mask in candidate_asserts:
#                 func_copy = copy.deepcopy(node)
#                 transformer = AssertionTransformer(
#                     target_lineno=assert_to_mask.lineno,
#                     target_col_offset=assert_to_mask.col_offset
#                 )
#                 masked_function_node = transformer.visit(func_copy)
                
#                 if transformer.transformed and transformer.ground_truth is not None:
#                     task_id_str = f"{reponame}_{task_id_gen.next()}"
                    
#                     # 获取质量分析信息
#                     quality_analysis = None
#                     if quality_filter:
#                         analysis = quality_filter.analyze_assertion_complexity(assert_to_mask)
#                         quality_analysis = analysis
                    
#                     data_entry = {
#                         "task_id": task_id_str,
#                         "reponame": reponame,
#                         "testpath": testpath.replace('\\', '/'),
#                         "testname": testname,
#                         "funcname": node.name,
#                         "imports": imports_list,
#                         "code": astunparse.unparse(node).strip(),
#                         "masked_code": astunparse.unparse(masked_function_node).strip(),
#                         "ground_truth": transformer.ground_truth
#                     }
                    
#                     # 添加质量分析信息
#                     if quality_analysis:
#                         data_entry["quality_analysis"] = quality_analysis
                    
#                     all_data_entries.append(data_entry)
                    
#     return all_data_entries

# def process_single_repo(repo_path: str, output_file: str):
#     """
#     遍历单个仓库，处理所有测试文件，并将结构化数据保存到JSONL文件中。
#     """
#     repo_abs_path = os.path.abspath(repo_path)
#     if not os.path.isdir(repo_abs_path):
#         print(f"错误: 仓库路径 '{repo_path}' 不存在或不是一个目录。已跳过。", file=sys.stderr)
#         return

#     task_id_gen = TaskIdGenerator()
    
#     print(f"\n--- 正在处理仓库: {os.path.basename(repo_abs_path)} ---")
#     all_processed_data = []
#     # 忽略常见的虚拟环境、Git和构建目录
#     excluded_dirs = {'.venv', 'venv', '.git', 'node_modules', 'build', 'dist', '__pycache__'}
    
#     for root, dirs, files in os.walk(repo_abs_path):
#         dirs[:] = [d for d in dirs if d not in excluded_dirs]
#         for file in files:
#             if file.startswith("test_") and file.endswith(".py"):
#                 file_path = os.path.join(root, file)
#                 processed_data = process_test_file(file_path, repo_abs_path, task_id_gen)
#                 all_processed_data.extend(processed_data)
    
#     if not all_processed_data:
#         print(f"在 {os.path.basename(repo_abs_path)} 中未找到带有等式断言的测试函数。")
#         return

#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for entry in all_processed_data:
#             json_line = json.dumps(entry, ensure_ascii=False)
#             f.write(json_line + '\n')
            
#     print(f"✅ 成功! 找到 {len(all_processed_data)} 个断言。数据集已保存至: {output_file}")

# def process_multiple_repos(repo_paths: list, output_dir: str):
#     """
#     协调处理多个仓库。
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"所有输出将被保存到: {os.path.abspath(output_dir)}")

#     for repo_path in repo_paths:
#         reponame = os.path.basename(os.path.abspath(repo_path))
#         output_file = os.path.join(output_dir, f"{reponame}.jsonl")
#         process_single_repo(repo_path, output_file)

# # ==============================================================================
# # 主逻辑 (融合了筛选条件)
# # ==============================================================================

# def main():
#     """
#     主函数，执行筛选和数据提取的融合流程。
#     """
#     current_dir = os.getcwd()
#     python_repos_dir = os.path.join(current_dir, 'python_repos')
#     output_results_dir = os.path.join(current_dir, 'output_results')
#     data_collection_dir = os.path.join(current_dir, 'data_collection')
    
#     # --- 阶段 1: 根据 report_functions.jsonl 的存在性来筛选仓库 ---
#     print("--- 阶段 1: 开始筛选要处理的仓库 ---")
    
#     # 检查所需目录是否存在
#     if not os.path.isdir(output_results_dir):
#         print(f"错误: 报告目录 '{output_results_dir}' 未找到。脚本中止。", file=sys.stderr)
#         return
#     if not os.path.isdir(python_repos_dir):
#         print(f"错误: 源码目录 '{python_repos_dir}' 未找到。脚本中止。", file=sys.stderr)
#         return

#     filtered_repo_names = []
#     # 假设 output_results 下的每个子目录对应一个仓库
#     repo_subdirs = [d for d in os.listdir(output_results_dir) if os.path.isdir(os.path.join(output_results_dir, d))]

#     for reponame in repo_subdirs:
#         report_file = os.path.join(output_results_dir, reponame, 'report_functions.jsonl')
        
#         # 核心筛选条件：报告文件必须存在
#         if os.path.exists(report_file):
#             # 并且，对应的代码仓库也必须存在
#             if os.path.isdir(os.path.join(python_repos_dir, reponame)):
#                 filtered_repo_names.append(reponame)
#                 print(f"找到报告 '{report_file}', 将仓库 '{reponame}' 添加到处理列表。")
#             else:
#                 print(f"警告: 找到了仓库 '{reponame}' 的报告文件, 但在 'python_repos' 目录中未找到其源代码。已跳过。")
#         else:
#             print(f"信息: 在仓库 '{reponame}' 中未找到报告文件，跳过处理。")


#     if not filtered_repo_names:
#         print("\n未找到任何满足条件的仓库（即同时存在报告文件和源代码）。脚本退出。")
#         return

#     print(f"\n✅ 筛选完成! 共找到 {len(filtered_repo_names)} 个仓库准备进行数据提取。")

#     # --- 阶段 2: 从筛选出的仓库中提取数据 ---
#     print("\n--- 阶段 2: 开始从筛选后的仓库中提取断言数据 ---")
    
#     # 构建需要处理的仓库的完整路径列表
#     repo_paths_to_process = [os.path.join(python_repos_dir, name) for name in filtered_repo_names]
    
#     # 调用数据提取流程
#     process_multiple_repos(repo_paths_to_process, data_collection_dir)
    
#     print("\n--- 所有任务已完成 ---")

# if __name__ == '__main__':
#     main()

# -*- coding: utf-8 -*-

import ast
import os
import json
import copy
import sys

# 尝试导入 astunparse，如果失败则提示用户安装
try:
    import astunparse
except ImportError:
    print("错误: 'astunparse' 库未安装。", file=sys.stderr)
    print("请使用以下命令进行安装: pip install astunparse", file=sys.stderr)
    sys.exit(1)

# 导入数据质量过滤器
try:
    from data_quality_filter import create_quality_filter
except ImportError:
    print("警告: 数据质量过滤器模块未找到，将使用基础过滤。", file=sys.stderr)
    create_quality_filter = None

# ==============================================================================
# [NEW] AST 访问器，用于查找测试并记录其类上下文
# ==============================================================================
class TestFinder(ast.NodeVisitor):
    """
    一个 AST 访问器，用于遍历代码树，找到所有测试函数
    并记录下它们自身以及它们所在的类（如果有的话）。
    """
    def __init__(self):
        self.found_tests = []
        self._current_class_name = None

    def visit_ClassDef(self, node: ast.ClassDef):
        # 进入一个类定义，记录下类名
        original_class_name = self._current_class_name
        self._current_class_name = node.name
        # 继续遍历类内部的节点
        self.generic_visit(node)
        # 离开类定义，恢复之前的上下文（支持嵌套类）
        self._current_class_name = original_class_name

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # 访问一个函数定义
        if node.name.startswith("test_"):
            # 如果是测试函数，则保存其节点和当前的类名
            self.found_tests.append({
                "node": node,
                "classname": self._current_class_name  # 如果不在类中，此值为 None
            })
        # 继续遍历函数内部的节点（虽然我们这里不需要）
        self.generic_visit(node)

# ==============================================================================
# 核心数据提取代码 (已修改以使用 TestFinder)
# ==============================================================================
class TaskIdGenerator:
    """一个简单的线程不安全的计数器，用于生成唯一的task ID。"""
    def __init__(self):
        self.current_id = 0
    
    def next(self) -> int:
        task_id = self.current_id
        self.current_id += 1
        return task_id

class AssertionTransformer(ast.NodeTransformer):
    """AST转换器，用于遮蔽断言语句的右侧。"""
    def __init__(self, target_lineno: int, target_col_offset: int):
        self.target_lineno = target_lineno
        self.target_col_offset = target_col_offset
        self.transformed = False
        self.ground_truth = None

    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        if not (node.lineno == self.target_lineno and node.col_offset == self.target_col_offset):
            return node
        if not (isinstance(node.test, ast.Compare) and
                node.test.ops and isinstance(node.test.ops[0], ast.Eq)):
            return node
        original_node = node.test.comparators[0]
        self.ground_truth = astunparse.unparse(original_node).strip()
        self.transformed = True
        try:
            node.test.comparators[0] = ast.Constant(value='???', kind=None)
        except AttributeError:
            node.test.comparators[0] = ast.Name(id='???', ctx=ast.Load())
        return node

def process_test_file(file_path: str, repo_root: str, task_id_gen: TaskIdGenerator):
    """
    解析一个 Python 测试文件，找到包含等式断言的测试函数，
    并为 *每一个* 这样的断言创建一条结构化数据。
    现在集成了数据质量过滤和类名捕获功能。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
    except (SyntaxError, UnicodeDecodeError, PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法解析 {file_path}。原因: {e}。已跳过。", file=sys.stderr)
        return []

    imports_list = [astunparse.unparse(node).strip() for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    all_data_entries = []
    reponame = os.path.basename(repo_root)
    testpath = os.path.relpath(file_path, repo_root)
    testname = os.path.basename(file_path)
    
    quality_filter = create_quality_filter() if create_quality_filter else None

    # [MODIFIED] 使用 TestFinder 代替 ast.walk
    finder = TestFinder()
    finder.visit(tree)

    for test_info in finder.found_tests:
        node = test_info["node"]
        classname = test_info["classname"] # 获取类名

        if quality_filter:
            quality_asserts = quality_filter.filter_test_function(node)
            candidate_asserts = [item["assert_node"] for item in quality_asserts]
        else:
            candidate_asserts = [sn for sn in ast.walk(node) if isinstance(sn, ast.Assert) and isinstance(sn.test, ast.Compare) and sn.test.ops and isinstance(sn.test.ops[0], ast.Eq)]
        
        for assert_to_mask in candidate_asserts:
            func_copy = copy.deepcopy(node)
            transformer = AssertionTransformer(assert_to_mask.lineno, assert_to_mask.col_offset)
            masked_function_node = transformer.visit(func_copy)
            
            if transformer.transformed and transformer.ground_truth is not None:
                task_id_str = f"{reponame}_{task_id_gen.next()}"
                quality_analysis = quality_filter.analyze_assertion_complexity(assert_to_mask) if quality_filter else None
                
                data_entry = {
                    "task_id": task_id_str,
                    "reponame": reponame,
                    "testpath": testpath.replace('\\', '/'),
                    "testname": testname,
                    "classname": classname,  # [NEW] 添加 classname 字段
                    "funcname": node.name,
                    "imports": imports_list,
                    "code": astunparse.unparse(node).strip(),
                    "masked_code": astunparse.unparse(masked_function_node).strip(),
                    "ground_truth": transformer.ground_truth
                }
                
                if quality_analysis:
                    data_entry["quality_analysis"] = quality_analysis
                
                all_data_entries.append(data_entry)
                
    return all_data_entries

# ... process_single_repo, process_multiple_repos 和 main 函数保持不变 ...
def process_single_repo(repo_path: str, output_file: str):
    repo_abs_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_abs_path):
        print(f"错误: 仓库路径 '{repo_path}' 不存在或不是一个目录。已跳过。", file=sys.stderr)
        return
    task_id_gen = TaskIdGenerator()
    print(f"\n--- 正在处理仓库: {os.path.basename(repo_abs_path)} ---")
    all_processed_data = []
    excluded_dirs = {'.venv', 'venv', '.git', 'node_modules', 'build', 'dist', '__pycache__'}
    for root, dirs, files in os.walk(repo_abs_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                processed_data = process_test_file(file_path, repo_abs_path, task_id_gen)
                all_processed_data.extend(processed_data)
    if not all_processed_data:
        print(f"在 {os.path.basename(repo_abs_path)} 中未找到带有等式断言的测试函数。")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_processed_data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"✅ 成功! 找到 {len(all_processed_data)} 个断言。数据集已保存至: {output_file}")

def process_multiple_repos(repo_paths: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有输出将被保存到: {os.path.abspath(output_dir)}")
    for repo_path in repo_paths:
        reponame = os.path.basename(os.path.abspath(repo_path))
        output_file = os.path.join(output_dir, f"{reponame}.jsonl")
        process_single_repo(repo_path, output_file)

def main():
    current_dir = os.getcwd()
    python_repos_dir = os.path.join(current_dir, 'python_repos')
    output_results_dir = os.path.join(current_dir, 'output_results')
    data_collection_dir = os.path.join(current_dir, 'data_collection')
    print("--- 阶段 1: 开始筛选要处理的仓库 ---")
    if not os.path.isdir(output_results_dir):
        print(f"错误: 报告目录 '{output_results_dir}' 未找到。脚本中止。", file=sys.stderr)
        return
    if not os.path.isdir(python_repos_dir):
        print(f"错误: 源码目录 '{python_repos_dir}' 未找到。脚本中止。", file=sys.stderr)
        return
    filtered_repo_names = []
    repo_subdirs = [d for d in os.listdir(output_results_dir) if os.path.isdir(os.path.join(output_results_dir, d))]
    for reponame in repo_subdirs:
        report_file = os.path.join(output_results_dir, reponame, 'report_functions.jsonl')
        if os.path.exists(report_file):
            if os.path.isdir(os.path.join(python_repos_dir, reponame)):
                filtered_repo_names.append(reponame)
                print(f"找到报告 '{report_file}', 将仓库 '{reponame}' 添加到处理列表。")
            else:
                print(f"警告: 找到了仓库 '{reponame}' 的报告文件, 但在 'python_repos' 目录中未找到其源代码。已跳过。")
        else:
            print(f"信息: 在仓库 '{reponame}' 中未找到报告文件，跳过处理。")
    if not filtered_repo_names:
        print("\n未找到任何满足条件的仓库（即同时存在报告文件和源代码）。脚本退出。")
        return
    print(f"\n✅ 筛选完成! 共找到 {len(filtered_repo_names)} 个仓库准备进行数据提取。")
    print("\n--- 阶段 2: 开始从筛选后的仓库中提取断言数据 ---")
    repo_paths_to_process = [os.path.join(python_repos_dir, name) for name in filtered_repo_names]
    process_multiple_repos(repo_paths_to_process, data_collection_dir)
    print("\n--- 所有任务已完成 ---")

if __name__ == '__main__':
    main()