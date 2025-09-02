# # align_datasets_final.py

# import os
# import json
# from collections import defaultdict
# import sys

# def group_data_by_function(file_path: str) -> dict:
#     """
#     读取一个.jsonl文件，并将其内容按 (testname, funcname) 分组。
#     """
#     data_map = defaultdict(list)
#     if not os.path.exists(file_path):
#         return data_map
        
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 # 使用 (testname, funcname) 作为唯一的函数标识符
#                 key = (data.get('testname'), data.get('funcname'))
#                 if key[0] and key[1]:
#                     data_map[key].append(data)
#             except (json.JSONDecodeError, KeyError):
#                 continue
#     return data_map

# def align_datasets_final():
#     """
#     根据重写前后每个测试函数中最小的断言数量来对齐数据集，
#     并强制统一 original 和 rewrite 数据的 task_id 和 testpath。
#     """
#     current_dir = os.getcwd()

#     # --- 定义目录路径 ---
#     source_dir_original = os.path.join(current_dir, 'data_collection')
#     source_dir_rewrite = os.path.join(current_dir, 'data_collection_from_rewrites_filtered')
    
#     align_base_dir = os.path.join(current_dir, 'data_collection_align')
#     output_dir_original = os.path.join(align_base_dir, 'original')
#     output_dir_rewrite = os.path.join(align_base_dir, 'rewrite')

#     # --- 准备工作 ---
#     print("--- 开始最终数据对齐（统一task_id和testpath） ---")
#     if not os.path.isdir(source_dir_original) or not os.path.isdir(source_dir_rewrite):
#         print(f"错误: 请确保源目录 'data_collection_filtered' 和 'data_collection_from_rewrites_filtered' 都存在。", file=sys.stderr)
#         return

#     os.makedirs(output_dir_original, exist_ok=True)
#     os.makedirs(output_dir_rewrite, exist_ok=True)
    
#     print(f"原始数据源: {source_dir_original}")
#     print(f"重写数据源: {source_dir_rewrite}")
#     print(f"对齐后输出至: {output_dir_original} 和 {output_dir_rewrite}")
#     print("-" * 70)

#     total_original_lines_before = 0
#     total_rewrite_lines_before = 0
#     total_aligned_lines = 0

#     # 以原始数据为基准进行遍历，确保文件列表一致
#     original_files = [f for f in os.listdir(source_dir_original) if f.endswith('.jsonl')]

#     for filename in original_files:
#         reponame = filename.replace('.jsonl', '')
#         print(f"\n--- 正在处理仓库: {reponame} ---")

#         original_file_path = os.path.join(source_dir_original, filename)
#         rewrite_file_path = os.path.join(source_dir_rewrite, filename)

#         if not os.path.exists(rewrite_file_path):
#             print(f"  警告: 在重写数据目录中未找到对应的文件 '{filename}'。已跳过。")
#             continue

#         # 1. 将两个源文件的数据按函数进行分组
#         original_data_map = group_data_by_function(original_file_path)
#         rewrite_data_map = group_data_by_function(rewrite_file_path)

#         original_line_count = sum(len(v) for v in original_data_map.values())
#         rewrite_line_count = sum(len(v) for v in rewrite_data_map.values())
#         total_original_lines_before += original_line_count
#         total_rewrite_lines_before += rewrite_line_count

#         print(f"  源文件行数: 原始={original_line_count}, 重写={rewrite_line_count}")

#         # 2. 对齐并写入文件
#         aligned_count_for_repo = 0
#         output_original_file = os.path.join(output_dir_original, filename)
#         output_rewrite_file = os.path.join(output_dir_rewrite, filename)

#         with open(output_original_file, 'w', encoding='utf-8') as f_out_orig, \
#              open(output_rewrite_file, 'w', encoding='utf-8') as f_out_rewrite:

#             common_keys = original_data_map.keys() & rewrite_data_map.keys()
#             print(f"  找到 {len(common_keys)} 个共同的测试函数进行对齐。")

#             for key in sorted(list(common_keys)):
#                 original_assertions = original_data_map[key]
#                 rewrite_assertions = rewrite_data_map[key]

#                 num_to_keep = min(len(original_assertions), len(rewrite_assertions))
                
#                 # --- 这是最核心的修正逻辑 ---
#                 for i in range(num_to_keep):
#                     original_item = original_assertions[i]
#                     rewrite_item = rewrite_assertions[i]

#                     # 创建一个 rewrite_item 的副本以进行修改
#                     aligned_rewrite_item = rewrite_item.copy()
                    
#                     # 强制将 rewrite item 的 task_id 和 testpath 与 original item 同步
#                     aligned_rewrite_item['task_id'] = original_item['task_id']
#                     aligned_rewrite_item['testpath'] = original_item['testpath']

#                     # 写入对齐后的一对数据
#                     f_out_orig.write(json.dumps(original_item, ensure_ascii=False) + '\n')
#                     f_out_rewrite.write(json.dumps(aligned_rewrite_item, ensure_ascii=False) + '\n')

#                 aligned_count_for_repo += num_to_keep
        
#         total_aligned_lines += aligned_count_for_repo
#         print(f"  ✅ 对齐完成: 两个输出文件均已写入 {aligned_count_for_repo} 行一一对应的数据。")

#     # --- 打印最终报告 ---
#     print("\n" + "=" * 70)
#     print("--- 总体对齐报告 ---")
#     print(f"\n对齐前总行数:")
#     print(f"  - 原始数据: {total_original_lines_before}")
#     print(f"  - 重写数据: {total_rewrite_lines_before}")
#     print(f"\n对齐后总行数:")
#     print(f"  - 'original' 和 'rewrite' 目录中的总行数均为: {total_aligned_lines}")
#     print("=" * 70)
#     print("\n--- 所有任务已完成 ---")


# if __name__ == '__main__':
#     align_datasets_final()

# align_datasets_final.py

import os
import json
from collections import defaultdict
import sys

def group_data_by_function(file_path: str) -> dict:
    """
    读取一个.jsonl文件，并将其内容按 (testname, classname, funcname) 分组。
    """
    data_map = defaultdict(list)
    if not os.path.exists(file_path):
        return data_map
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                key = (
                    data.get('testname'),
                    data.get('classname'), 
                    data.get('funcname')
                )
                if key[0] and key[2]:
                    data_map[key].append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return data_map

def align_datasets_final():
    """
    根据重写前后每个测试函数中最小的断言数量来对齐数据集，
    并强制统一 original 和 rewrite 数据的 task_id 和 testpath。
    """
    current_dir = os.getcwd()

    # --- 定义目录路径 ---
    source_dir_original = os.path.join(current_dir, 'data_collection')
    source_dir_rewrite = os.path.join(current_dir, 'data_collection_from_rewrites_filtered')
    
    align_base_dir = os.path.join(current_dir, 'data_collection_align')
    output_dir_original = os.path.join(align_base_dir, 'original')
    output_dir_rewrite = os.path.join(align_base_dir, 'rewrite')

    # --- 准备工作 ---
    print("--- 开始最终数据对齐（统一task_id和testpath） ---")
    if not os.path.isdir(source_dir_original) or not os.path.isdir(source_dir_rewrite):
        print(f"错误: 请确保源目录 'data_collection' 和 'data_collection_from_rewrites_filtered' 都存在。", file=sys.stderr)
        return

    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_rewrite, exist_ok=True)
    
    print(f"原始数据源: {source_dir_original}")
    print(f"重写数据源: {source_dir_rewrite}")
    print(f"对齐后输出至: {output_dir_original} 和 {output_dir_rewrite}")
    print("-" * 70)

    total_original_lines_before, total_rewrite_lines_before, total_aligned_lines = 0, 0, 0
    original_files = [f for f in os.listdir(source_dir_original) if f.endswith('.jsonl')]

    for filename in original_files:
        reponame = filename.replace('.jsonl', '')
        print(f"\n--- 正在处理仓库: {reponame} ---")

        original_file_path = os.path.join(source_dir_original, filename)
        rewrite_file_path = os.path.join(source_dir_rewrite, filename)

        if not os.path.exists(rewrite_file_path):
            print(f"  警告: 在重写数据目录中未找到对应的文件 '{filename}'。已跳过。"); continue

        original_data_map = group_data_by_function(original_file_path)
        rewrite_data_map = group_data_by_function(rewrite_file_path)

        original_line_count = sum(len(v) for v in original_data_map.values())
        rewrite_line_count = sum(len(v) for v in rewrite_data_map.values())
        total_original_lines_before += original_line_count
        total_rewrite_lines_before += rewrite_line_count

        print(f"  源文件行数: 原始={original_line_count}, 重写={rewrite_line_count}")

        aligned_count_for_repo = 0
        output_original_file, output_rewrite_file = os.path.join(output_dir_original, filename), os.path.join(output_dir_rewrite, filename)

        with open(output_original_file, 'w', encoding='utf-8') as f_out_orig, \
             open(output_rewrite_file, 'w', encoding='utf-8') as f_out_rewrite:

            common_keys_set = original_data_map.keys() & rewrite_data_map.keys()
            
            # --- [CRITICAL FIX] ---
            # 在排序前，使用一个 lambda 函数作为 key，将元组中的所有元素都转换为字符串。
            # 这样 None 就会变成 'None'，就可以安全地与其他字符串进行比较了。
            common_keys = sorted(
                list(common_keys_set), 
                key=lambda k: (str(k[0]), str(k[1]), str(k[2]))
            )
            # --- [END CRITICAL FIX] ---

            print(f"  找到 {len(common_keys)} 个共同的测试函数/方法进行对齐。")

            for key in common_keys:
                original_assertions, rewrite_assertions = original_data_map[key], rewrite_data_map[key]
                num_to_keep = min(len(original_assertions), len(rewrite_assertions))
                
                for i in range(num_to_keep):
                    original_item, rewrite_item = original_assertions[i], rewrite_assertions[i]
                    aligned_rewrite_item = rewrite_item.copy()
                    aligned_rewrite_item['task_id'] = original_item['task_id']
                    aligned_rewrite_item['testpath'] = original_item['testpath']
                    f_out_orig.write(json.dumps(original_item, ensure_ascii=False) + '\n')
                    f_out_rewrite.write(json.dumps(aligned_rewrite_item, ensure_ascii=False) + '\n')
                aligned_count_for_repo += num_to_keep
        
        total_aligned_lines += aligned_count_for_repo
        print(f"  ✅ 对齐完成: 两个输出文件均已写入 {aligned_count_for_repo} 行一一对应的数据。")

    print("\n" + "=" * 70 + "\n--- 总体对齐报告 ---")
    print(f"\n对齐前总行数:\n  - 原始数据: {total_original_lines_before}\n  - 重写数据: {total_rewrite_lines_before}")
    print(f"\n对齐后总行数:\n  - 'original' 和 'rewrite' 目录中的总行数均为: {total_aligned_lines}")
    print("=" * 70 + "\n\n--- 所有任务已完成 ---")

if __name__ == '__main__':
    align_datasets_final()