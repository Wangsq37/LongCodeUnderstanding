# import json
# import os
# from pathlib import Path
# import time
# import datetime
# import tiktoken
# from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
# import random

# # ==============================================================================
# # 实验配置
# # ==============================================================================

# # --- API 和模型配置 ---
# CUSTOM_API_KEY = "sk-Eyhl41vT8nBB8xLY77D7C9Bf0198453dA2797221Aa75E486"
# CUSTOM_API_BASE_URL = "https://api.vveai.com/v1/"
# OPENAI_COMPATIBLE_MODEL_NAME = "gpt-4.1-mini"

# TOKENIZER_NAME = "o200k_base"

# # --- 核心目录配置 ---
# BASE_DATA_DIR = Path("data_collection_align")
# BM25_RANK_DIR = Path("output_with_bm25_rank")
# REPO_BASE_DIR = Path("python_repos")
# EXPERIMENTS_OUTPUT_DIR = Path("experiments_output_gpt-4.1-mini_30k")

# # --- 实验参数 ---
# MODEL_CONTEXT_WINDOW = 30720
# MAX_ITEMS_PER_REPO = 200
# NUM_SAMPLES_PER_TASK = 5
# MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL = 20
# NUM_CONFUSION_FILES_TO_SAMPLE = 20
# TOP_N_TO_EXCLUDE_FOR_CONFUSION = 20
# TEMPERATURE = 0.7
# TOP_P = 0.9
# RETRY_DELAY_SECONDS = 5

# # --- 派生常量 ---
# MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = MODEL_CONTEXT_WINDOW - 100
# SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 50

# # ==============================================================================
# # Prompt模板
# # ==============================================================================
# PROMPT_TEMPLATE_PARTS = {
# "system_message": """Your task is to infer the correct values for the '???' placeholder based on provided Python code.
# # Rules
# 1.  Analyze the provided code and related files to determine the value.
# 2.  The value MUST be a determined value or a valid Python expression.
# 3.  **Do not** output any explanations, reasoning, introductions, or any other text outside of the specified output format. Your output must be clean.
# # Output format

# ```output
# answer_of_placeholder
# ```

# # Example
# Here is an example of a perfect interaction.

# ## Related files are as follow:
# ### file: config.py
# ```python
# DEFAULT_TIMEOUT = 50
# ```

# ### file: get_timeout.py
# ```python
# from config import DEFAULT_TIMEOUT
# def get_timeout():
#     return DEFAULT_TIMEOUT
# ```

# ## Test function is as follows:
# ### test
# ```python
# def test_timeout():
#     from get_timeout import get_timeout
#     assert get_timeout() == '???'
# ```

# ### Your Correct Output:
# ```output
# 50
# ```
# """,
# "user_prompt_template": """## Related files are as follows:
# {related_files_content}

# ## Test function is as follows:
# ```python
# {masked_code_content}
# ```

# Based on all the provided context and the test function, infer the correct value or expression for the "???" placeholder. Output it directly in the required format. Do not output extra information.
# """
# }


# # ==============================================================================
# # 初始化和辅助函数
# # ==============================================================================
# random.seed(42)
# try:
#     tokenizer_encoding = tiktoken.get_encoding(TOKENIZER_NAME)
# except KeyError:
#     print(f"错误: Tokenizer编码 '{TOKENIZER_NAME}' 未找到。"); exit(1)
# try:
#     client = OpenAI(api_key=CUSTOM_API_KEY, base_url=CUSTOM_API_BASE_URL, timeout=60.0, max_retries=2)
# except Exception as e:
#     print(f"错误: 初始化OpenAI客户端失败: {e}"); exit(1)

# def count_tokens_custom(text, encoding):
#     return len(encoding.encode(text))

# def load_jsonl(file_path):
#     data = []
#     if not Path(file_path).exists():
#         return data
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for ln, line in enumerate(f):
#             try: data.append(json.loads(line))
#             except json.JSONDecodeError as e: print(f"警告: 在 {file_path} 第{ln+1}行JSON解码错误: {e}。行内容: '{line.strip()}'")
#     return data

# def get_related_files_content_budgeted_custom(reponame_arg, related_files_rank_list, current_encoding, token_budget_for_related_files):
#     content_parts, accumulated_tokens = [], 0
#     if token_budget_for_related_files <= 0: return "# 没有为相关文件分配token预算。\n", 0
    
#     for rel_path_str in related_files_rank_list:
#         if accumulated_tokens >= token_budget_for_related_files: break
#         file_path = REPO_BASE_DIR / reponame_arg / rel_path_str
#         file_header_str = f"### file: {rel_path_str}\n```python\n"
#         file_footer_str = "\n```\n"
#         file_header_tokens = count_tokens_custom(file_header_str, current_encoding)
#         file_footer_tokens = count_tokens_custom(file_footer_str, current_encoding)
        
#         file_content_str, error_reading = "", False
#         if file_path.exists() and file_path.is_file():
#             try:
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content_str = f.read()
#             except Exception as e: file_content_str, error_reading = f"# 读取文件错误 {file_path}: {e}", True
#         else: file_content_str, error_reading = f"# 文件 {file_path} 未找到", True
            
#         available_tokens_for_this_file = token_budget_for_related_files - accumulated_tokens
#         if file_header_tokens + file_footer_tokens >= available_tokens_for_this_file: break
        
#         current_file_str_parts = [file_header_str]
#         current_file_tokens = file_header_tokens

#         content_tokens_budget = available_tokens_for_this_file - file_header_tokens - file_footer_tokens
#         if content_tokens_budget > 0:
#             content_token_ids = current_encoding.encode(file_content_str)
#             if len(content_token_ids) > content_tokens_budget:
#                 truncated_content_token_ids = content_token_ids[:content_tokens_budget]
#                 try: truncated_content_str = current_encoding.decode(truncated_content_token_ids)
#                 except: truncated_content_str = "[内容截断 - 解码错误]"
#                 current_file_str_parts.append(truncated_content_str)
#                 current_file_tokens += count_tokens_custom(truncated_content_str, current_encoding)
#             else:
#                 current_file_str_parts.append(file_content_str)
#                 current_file_tokens += len(content_token_ids)

#         current_file_str_parts.append(file_footer_str)
#         current_file_tokens += file_footer_tokens

#         if accumulated_tokens + current_file_tokens <= token_budget_for_related_files:
#             content_parts.append("".join(current_file_str_parts))
#             accumulated_tokens += current_file_tokens
#         else:
#             break
            
#     if not content_parts:
#         final_str = "# 没有相关文件内容可以在token限制内被包含。\n"
#         return final_str, count_tokens_custom(final_str, current_encoding)
#     return "".join(content_parts), accumulated_tokens

# # [新增] 新的辅助函数，用于清理和去重JSONL文件
# def deduplicate_and_rewrite_jsonl(file_path: Path):
#     """
#     读取一个JSONL文件，根据 (task_id, sample_id) 去重，只保留最新的条目，
#     然后用去重后的内容重写该文件。
#     """
#     if not file_path.exists():
#         return

#     # 使用字典来存储每个任务样本的最新记录
#     # key: (task_id, sample_id), value: 完整的JSON对象
#     latest_records = {}
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 # 必须有 task_id 和 sample_id 才能进行去重
#                 if 'task_id' in data and 'sample_id' in data:
#                     key = (data['task_id'], data['sample_id'])
#                     latest_records[key] = data
#             except json.JSONDecodeError:
#                 # 忽略格式不正确的行
#                 continue

#     # 用去重后并保留最新的记录重写文件
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for record in latest_records.values():
#             f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
#     print(f"   已清理并去重文件: {file_path}，保留了 {len(latest_records)} 条最新记录。")


# # ==============================================================================
# # 核心实验逻辑
# # ==============================================================================
# def process_repository(reponame, dataset_dir, base_predictions_dir, base_interactions_dir):
#     print(f"\n--- 正在处理仓库: {reponame} ---")
#     dataset_file = dataset_dir / f"{reponame}.jsonl"
#     bm25_file = BM25_RANK_DIR / f"{reponame}.jsonl"

#     if not dataset_file.exists(): print(f"数据集文件未找到: {dataset_file}。"); return
#     if not bm25_file.exists(): print(f"BM25排名文件未找到: {bm25_file}。"); return

#     test_data_list = load_jsonl(dataset_file)[:MAX_ITEMS_PER_REPO]
#     print(f"   已加载 {len(test_data_list)} 个条目 (最大 {MAX_ITEMS_PER_REPO})。")
#     bm25_lookup = {item['task_id']: item for item in load_jsonl(bm25_file)}
    
#     tokens_system_message = count_tokens_custom(PROMPT_TEMPLATE_PARTS["system_message"], tokenizer_encoding)
#     static_user_prompt_template = PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content="", masked_code_content="")
#     tokens_static_user_prompt = count_tokens_custom(static_user_prompt_template, tokenizer_encoding)

#     # for experiment_type in ["retrieval", "confusion"]:
#     for experiment_type in ["retrieval"]:
#         print(f"   实验模式: {experiment_type}")
#         current_preds_dir = base_predictions_dir / experiment_type
#         current_inters_dir = base_interactions_dir / experiment_type
#         current_preds_dir.mkdir(parents=True, exist_ok=True)
#         current_inters_dir.mkdir(parents=True, exist_ok=True)
#         predictions_output_file = current_preds_dir / f"{reponame}.jsonl"
#         interactions_output_file = current_inters_dir / f"{reponame}.jsonl"

#         # [修改] 在开始处理前，先清理输出文件，移除旧的/失败的尝试
#         deduplicate_and_rewrite_jsonl(predictions_output_file)
#         deduplicate_and_rewrite_jsonl(interactions_output_file)

#         # --- 断点续传逻辑 ---
#         completed_tasks = set()
#         # 现在的加载逻辑是基于清理过的文件，所以更准确
#         existing_preds = load_jsonl(predictions_output_file)
#         for pred in existing_preds:
#             # 这里的逻辑保持不变，因为文件已经是干净的了
#             if 'task_id' in pred and 'sample_id' in pred:
#                 if pred.get('finish_reason') not in ['api_error', 'sdk_error', 'init_fail']:
#                     completed_tasks.add((pred['task_id'], pred['sample_id']))
#         print(f"   已加载 {len(completed_tasks)} 个已成功完成的样本。将跳过这些任务。")

#         # 以追加模式打开文件，因为它们已经被清理过了
#         with open(predictions_output_file, 'a', encoding='utf-8') as pred_f, \
#              open(interactions_output_file, 'a', encoding='utf-8') as intr_f:
#             for item_idx, test_case in enumerate(test_data_list):
#                 task_id = test_case['task_id']
#                 if task_id not in bm25_lookup: 
#                     print(f"     警告: 任务ID {task_id} 没有BM25排名。跳过。")
#                     continue
                
#                 all_ranked_files = bm25_lookup[task_id].get('related_files_rank', [])
#                 related_files_for_experiment = []
#                 if experiment_type == "retrieval": related_files_for_experiment = all_ranked_files[:MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL]
#                 elif experiment_type == "confusion":
#                     candidates = all_ranked_files[TOP_N_TO_EXCLUDE_FOR_CONFUSION:]
#                     if len(candidates) >= NUM_CONFUSION_FILES_TO_SAMPLE: related_files_for_experiment = random.sample(candidates, NUM_CONFUSION_FILES_TO_SAMPLE)
#                     else: related_files_for_experiment = candidates
#                 random.shuffle(related_files_for_experiment)
                
#                 for sample_idx in range(NUM_SAMPLES_PER_TASK):
#                     task_key = (task_id, sample_idx)
#                     if task_key in completed_tasks:
#                         continue

#                     print(f"     处理任务ID: {task_id} ({item_idx+1}/{len(test_data_list)}), 样本 {sample_idx + 1}/{NUM_SAMPLES_PER_TASK}")
#                     masked_code = test_case['masked_code']
#                     tokens_test_code = count_tokens_custom(masked_code, tokenizer_encoding)
                    
#                     token_budget_for_dynamic_content = (
#                         MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - tokens_system_message -
#                         tokens_static_user_prompt - SAFETY_MARGIN_TOKENS_PROMPT_BUILDING
#                     )
#                     token_budget_for_related_files = token_budget_for_dynamic_content - tokens_test_code
                    
#                     related_files_str, _ = get_related_files_content_budgeted_custom(
#                         reponame, related_files_for_experiment, 
#                         tokenizer_encoding, token_budget_for_related_files
#                     ) if token_budget_for_related_files > 0 and related_files_for_experiment else ("# 无相关文件上下文。\n", 0)
                    
#                     user_prompt = PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(
#                         related_files_content=related_files_str,
#                         masked_code_content=masked_code
#                     )
                    
#                     messages = [{"role": "system", "content": PROMPT_TEMPLATE_PARTS["system_message"]}, {"role": "user", "content": user_prompt}]
                    
#                     response_text, finish_reason, usage, response_obj, error_details = \
#                         "ERROR: INIT", "init_fail", None, None, None
                    
#                     while True:
#                         try:
#                             response = client.chat.completions.create(
#                                 model=OPENAI_COMPATIBLE_MODEL_NAME, 
#                                 messages=messages, 
#                                 temperature=TEMPERATURE, 
#                                 top_p=TOP_P, 
#                             )
#                             response_obj = response
#                             response_text = response.choices[0].message.content
#                             finish_reason = response.choices[0].finish_reason
#                             usage = response.usage.model_dump() if response.usage else None
                            
#                             print(f"         API调用成功。Finish reason: {finish_reason}")
#                             break

#                         except (APITimeoutError, APIConnectionError, RateLimitError) as e:
#                             print(f"         API可重试错误 ({e.__class__.__name__})。将在 {RETRY_DELAY_SECONDS} 秒后重试...")
#                             time.sleep(RETRY_DELAY_SECONDS)

#                         except APIError as e:
#                             if hasattr(e, 'status_code') and e.status_code in [500, 502, 503, 504]:
#                                 print(f"         API服务器端错误 (可重试, status={e.status_code})。将在 {RETRY_DELAY_SECONDS} 秒后重试...")
#                                 time.sleep(RETRY_DELAY_SECONDS)
#                             else:
#                                 print(f"         API错误 (不可重试): {e}")
#                                 response_text = f"ERROR: APIError - {e.__class__.__name__}"
#                                 finish_reason = "api_error"
#                                 error_details = {"type": e.__class__.__name__, "message": str(e)}
#                                 if hasattr(e, 'status_code'):
#                                     error_details['status_code'] = e.status_code
#                                 if hasattr(e, 'body'):
#                                     error_details['raw_body'] = str(e.body) if e.body else None
#                                 break 

#                         except Exception as e:
#                             print(f"         意外错误 (不可重试): {e}")
#                             response_text = f"ERROR: SDK/Network Error - {e.__class__.__name__}"
#                             finish_reason = "sdk_error"
#                             error_details = {"type": e.__class__.__name__, "message": str(e)}
#                             break 

#                     intr_f.write(json.dumps({
#                         "task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx,
#                         "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
#                         "messages_sent": messages, "model_used": OPENAI_COMPATIBLE_MODEL_NAME,
#                         "related_files_provided_to_prompt": related_files_for_experiment,
#                         "api_response_object_str": str(response_obj), "api_error_details": error_details
#                     }, ensure_ascii=False) + "\n")
#                     pred_f.write(json.dumps({
#                         "task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, 
#                         "response": response_text.strip() if isinstance(response_text, str) else str(response_text), 
#                         "finish_reason": finish_reason, "usage": usage
#                     }, ensure_ascii=False) + "\n")
                    
#                     pred_f.flush()
#                     intr_f.flush()
                    
#                     time.sleep(0.3)
#     print(f"--- 完成仓库: {reponame} ---")

# # ==============================================================================
# # 主函数
# # ==============================================================================
# def main():
#     data_conditions = {"original": BASE_DATA_DIR / "original", "rewrite": BASE_DATA_DIR / "rewrite"}
#     for condition_name, dataset_dir in data_conditions.items():
#         print("\n" + "="*80)
#         print(f"开始实验条件: '{condition_name.upper()}'")
#         print(f"数据源: {dataset_dir}")
#         print("="*80)
#         if not dataset_dir.exists(): 
#             print(f"错误: 数据目录 '{dataset_dir}' 不存在。跳过此实验条件。")
#             continue
        
#         base_predictions_dir = EXPERIMENTS_OUTPUT_DIR / condition_name / "predictions"
#         base_interactions_dir = EXPERIMENTS_OUTPUT_DIR / condition_name / "interactions"
#         base_predictions_dir.mkdir(parents=True, exist_ok=True)
#         base_interactions_dir.mkdir(parents=True, exist_ok=True)
        
#         repo_names = sorted([p.stem for p in dataset_dir.glob("*.jsonl")])
#         if not repo_names: 
#             print(f"在 {dataset_dir} 中未找到.jsonl文件。")
#             continue
        
#         print(f"找到的仓库: {repo_names}")
#         for reponame in repo_names:
#             process_repository(reponame, dataset_dir, base_predictions_dir, base_interactions_dir)
#     print("\n--- 所有实验条件均已完成 ---")

# if __name__ == "__main__":
#     main()


# import json
# import os
# from pathlib import Path
# import time
# import datetime
# import random
# import re
# import argparse
# from tqdm import tqdm
# from openai import OpenAI, APIError
# import tiktoken

# # ==============================================================================
# # 实验配置
# # ==============================================================================
# class Config:
#     # --- 动态API配置 (由命令行参数设置) ---
#     API_BASE_URL = None
#     API_KEY = None
#     MODEL_NAME = None
#     MODEL_CONTEXT_WINDOW = None
    
#     # --- 静态目录配置 ---
#     BASE_DATA_DIR = Path("data_collection_align")
#     BM25_RANK_DIR = Path("output_with_bm25_rank")
#     CALL_CHAIN_DATA_DIR = Path("output_results")
#     REPO_BASE_DIR = Path("python_repos")
#     EXPERIMENTS_OUTPUT_DIR = None

#     # --- 实验参数 ---
#     MAX_COMPLETION_TOKENS = 512
#     MAX_ITEMS_PER_REPO = 200
#     NUM_SAMPLES_PER_TASK = 5
#     TEMPERATURE = 0.7
#     TOP_P = 0.9
#     REQUEST_DELAY_SECONDS = 1 # 避免API限速

#     # --- Tokenizer ---
#     TOKENIZER_NAME = "o200k_base"

#     # --- 不同实验类型的文件选择参数 ---
#     MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL = 20
#     NUM_CONFUSION_FILES_TO_SAMPLE = 20
#     TOP_N_TO_EXCLUDE_FOR_CONFUSION = 20
#     MAX_ORACLE_FILES_TO_CONSIDER = 20

#     # --- 派生常量 ---
#     MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
#     SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 50

#     PROMPT_TEMPLATE_PARTS = {
# "system_message": """Your task is to infer the correct values for the '???' placeholder based on provided Python code.
# # Rules
# 1.  Analyze the provided code and related files to determine the value.
# 2.  The value MUST be a determined value or a valid Python expression.
# 3.  **Do not** output any explanations, reasoning, introductions, or any other text outside of the specified output format. Your output must be clean.
# # Output format

# ```output
# answer_of_placeholder
# ```

# # Example
# Here is an example of a perfect interaction.

# ## Related files are as follow:
# ### file: config.py
# ```python
# DEFAULT_TIMEOUT = 50
# ```

# ### file: get_timeout.py
# ```python
# from config import DEFAULT_TIMEOUT
# def get_timeout():
#     return DEFAULT_TIMEOUT
# ```

# ## Test function is as follows:
# ### test
# ```python
# def test_timeout():
#     from get_timeout import get_timeout
#     assert get_timeout() == '???'
# ```

# ### Your Correct Output:
# ```output
# 50
# ```
# """,
#     "user_prompt_template": """## Related files are as follows:
# {related_files_content}

# ## Test function is as follows:
# ```python
# {masked_code_content}
# ```

# Based on all the provided context and the test function, infer the correct value or expression for the "???" placeholder. Output it directly in the required format. Do not output extra information.
# """
#     }

# # ==============================================================================
# # 辅助函数
# # ==============================================================================
# def count_tokens_custom(text, tok):
#     return len(tok.encode(text))

# def load_jsonl(file_path):
#     data = []
#     if not file_path.exists(): return data
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for ln, line in enumerate(f):
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 pass # Silently skip malformed lines
#     return data

# def get_related_files_content_budgeted_custom(reponame_arg, related_files_rank_list, current_tokenizer, token_budget_for_related_files):
#     content_parts, accumulated_tokens = [], 0
#     if token_budget_for_related_files <= 0: return "# 没有为相关文件分配token预算。\n", 0
#     for rel_path_str in related_files_rank_list:
#         if accumulated_tokens >= token_budget_for_related_files: break
#         file_path = Config.REPO_BASE_DIR / reponame_arg / rel_path_str
#         file_header_str = f"### file: {rel_path_str}\n```python\n"
#         file_footer_str = "\n```\n"
#         file_header_tokens = count_tokens_custom(file_header_str, current_tokenizer)
#         file_footer_tokens = count_tokens_custom(file_footer_str, current_tokenizer)
        
#         file_content_str = ""
#         try:
#             if file_path.exists() and file_path.is_file():
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content_str = f.read()
#             else: file_content_str = f"# 文件 {file_path} 未找到"
#         except Exception as e: file_content_str = f"# 读取文件错误 {file_path}: {e}"
            
#         available_tokens_for_this_file = token_budget_for_related_files - accumulated_tokens
#         if file_header_tokens + file_footer_tokens >= available_tokens_for_this_file: break
        
#         current_file_str_parts = [file_header_str]
#         content_tokens_budget = available_tokens_for_this_file - file_header_tokens - file_footer_tokens
        
#         if content_tokens_budget > 0:
#             content_token_ids = current_tokenizer.encode(file_content_str)
#             if len(content_token_ids) > content_tokens_budget:
#                 truncated_content_token_ids = content_token_ids[:content_tokens_budget]
#                 truncated_content_str = current_tokenizer.decode(truncated_content_token_ids)
#                 current_file_str_parts.append(truncated_content_str)
#             else:
#                 current_file_str_parts.append(file_content_str)
#         current_file_str_parts.append(file_footer_str)
        
#         full_part = "".join(current_file_str_parts)
#         part_tokens = count_tokens_custom(full_part, current_tokenizer)
        
#         if accumulated_tokens + part_tokens <= token_budget_for_related_files:
#             content_parts.append(full_part)
#             accumulated_tokens += part_tokens
#         else: break
            
#     if not content_parts:
#         return "# 没有相关文件内容可以在token限制内被包含。\n", 0
#     return "".join(content_parts), accumulated_tokens

# def normalize_funcname(name: str) -> str:
#     name = re.sub(r'\[.*\]$', '', name)
#     if '::' in name:
#         name = name.split('::')[-1]
#     return name

# # ==============================================================================
# # 核心实验逻辑
# # ==============================================================================
# def process_repository(reponame, dataset_dir, condition_name, client, tokenizer):
#     dataset_file = dataset_dir / f"{reponame}.jsonl"
#     bm25_file = Config.BM25_RANK_DIR / f"{reponame}.jsonl"
#     call_chain_file = Config.CALL_CHAIN_DATA_DIR / reponame / "report_functions.jsonl"
#     if not dataset_file.exists(): return

#     test_data_list = load_jsonl(dataset_file)[:Config.MAX_ITEMS_PER_REPO]
#     bm25_lookup = {item['task_id']: item for item in load_jsonl(bm25_file)}
#     call_chain_data = load_jsonl(call_chain_file)
#     call_chain_lookup = {}
#     for item in call_chain_data:
#         dirty_func_name = item.get('test_function')
#         if not dirty_func_name: continue
#         normalized_name = normalize_funcname(dirty_func_name)
#         sorted_deps = sorted(item.get('dependencies', []), key=lambda x: x.get('hops', float('inf')))
#         call_chain_lookup[normalized_name] = [dep['file'] for dep in sorted_deps]

#     tokens_system_message = count_tokens_custom(Config.PROMPT_TEMPLATE_PARTS["system_message"], tokenizer)
#     static_user_prompt_template = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content="", masked_code_content="")
#     tokens_static_user_prompt = count_tokens_custom(static_user_prompt_template, tokenizer)

#     # for experiment_type in ["retrieval", "confusion", "oracle"]:
#     for experiment_type in ["oracle"]:
#         print(f"    -> 开始实验: {experiment_type}")
#         current_preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "predictions" / experiment_type
#         current_inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "interactions" / experiment_type
#         current_preds_dir.mkdir(parents=True, exist_ok=True)
#         current_inters_dir.mkdir(parents=True, exist_ok=True)
#         predictions_output_file = current_preds_dir / f"{reponame}.jsonl"
#         interactions_output_file = current_inters_dir / f"{reponame}.jsonl"

#         with open(predictions_output_file, 'w', encoding='utf-8') as pred_f, \
#              open(interactions_output_file, 'w', encoding='utf-8') as intr_f:
            
#             tasks_to_process = []
#             for test_case in test_data_list:
#                 task_id = test_case['task_id']
#                 related_files_for_experiment = []

#                 if experiment_type == "retrieval":
#                     if task_id in bm25_lookup:
#                         files = bm25_lookup[task_id].get('related_files_rank', [])
#                         related_files_for_experiment = files[:Config.MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL]
#                         random.shuffle(related_files_for_experiment)
#                 elif experiment_type == "confusion":
#                     if task_id in bm25_lookup:
#                         files = bm25_lookup[task_id].get('related_files_rank', [])
#                         pool = files[Config.TOP_N_TO_EXCLUDE_FOR_CONFUSION:]
#                         related_files_for_experiment = random.sample(pool, min(len(pool), Config.NUM_CONFUSION_FILES_TO_SAMPLE))
#                 elif experiment_type == "oracle":
#                     func_name = test_case.get('funcname')
#                     if func_name and normalize_funcname(func_name) in call_chain_lookup:
#                         files = call_chain_lookup[normalize_funcname(func_name)]
#                         related_files_for_experiment = files[:Config.MAX_ORACLE_FILES_TO_CONSIDER]
                
#                 for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
#                     tasks_to_process.append({"test_case": test_case, "related_files": related_files_for_experiment, "sample_idx": sample_idx})
            
#             task_iterator = tqdm(tasks_to_process, desc=f"      Tasks", leave=False, position=1)
#             for task_item in task_iterator:
#                 test_case, related_files, sample_idx = task_item["test_case"], task_item["related_files"], task_item["sample_idx"]
#                 task_id = test_case["task_id"]

#                 masked_code = test_case['masked_code']
#                 tokens_test_code = count_tokens_custom(masked_code, tokenizer)
#                 token_budget_for_dynamic_content = (Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - tokens_system_message - tokens_static_user_prompt - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING)
#                 token_budget_for_related_files = token_budget_for_dynamic_content - tokens_test_code
                
#                 related_files_str, _ = get_related_files_content_budgeted_custom(reponame, related_files, tokenizer, token_budget_for_related_files) if token_budget_for_related_files > 0 else ("# 无相关文件上下文。\n", 0)
                
#                 user_prompt = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content=related_files_str, masked_code_content=masked_code)
#                 messages = [{"role": "system", "content": Config.PROMPT_TEMPLATE_PARTS["system_message"]}, {"role": "user", "content": user_prompt}]
                
#                 response_text, finish_reason, usage, error_details = "ERROR: INIT", "error", None, None
#                 try:
#                     response = client.chat.completions.create(
#                         model=Config.MODEL_NAME, messages=messages, temperature=Config.TEMPERATURE,
#                         top_p=Config.TOP_P, max_tokens=Config.MAX_COMPLETION_TOKENS
#                     )
#                     response_text = response.choices[0].message.content
#                     finish_reason = response.choices[0].finish_reason
#                     usage = response.usage.model_dump() if response.usage else None
#                 except APIError as e:
#                     response_text, error_details = f"ERROR: APIError - {e.message}", {"type": type(e).__name__, "message": str(e), "status_code": e.status_code}
#                 except Exception as e:
#                     response_text, error_details = f"ERROR: SDK/Network Error - {type(e).__name__}", {"type": type(e).__name__, "message": str(e)}

#                 intr_f.write(json.dumps({"task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "messages_sent_to_model": messages, "model_used": Config.MODEL_NAME, "related_files_provided_to_prompt": related_files, "api_error_details": error_details}, ensure_ascii=False) + "\n")
#                 pred_f.write(json.dumps({"task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, "response": response_text.strip(), "finish_reason": finish_reason, "usage": usage}, ensure_ascii=False) + "\n")
#                 time.sleep(Config.REQUEST_DELAY_SECONDS)

# # ==============================================================================
# # 主函数 (协调器)
# # ==============================================================================
# def main():

#     CUSTOM_API_KEY = "sk-Eyhl41vT8nBB8xLY77D7C9Bf0198453dA2797221Aa75E486"
#     CUSTOM_API_BASE_URL = "https://api.vveai.com/v1/"
#     OPENAI_COMPATIBLE_MODEL_NAME = "gpt-4.1-mini"

    
#     Config.API_BASE_URL = CUSTOM_API_BASE_URL 
#     Config.API_KEY = CUSTOM_API_KEY
#     Config.MODEL_NAME = OPENAI_COMPATIBLE_MODEL_NAME
#     Config.MODEL_CONTEXT_WINDOW = 10240
#     model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.MODEL_NAME)
#     Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_output_oracle_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}")
#     Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - 100
    
#     # 当前实验为gpt-4.1min-10k的oracle实验


#     print("="*80)
#     print("实验配置初始化完成")
#     print(f"  - API Base URL: {Config.API_BASE_URL}")
#     print(f"  - Model Name: {Config.MODEL_NAME}")
#     print(f"  - 上下文窗口: {Config.MODEL_CONTEXT_WINDOW}")
#     print(f"  - 输出目录: {Config.EXPERIMENTS_OUTPUT_DIR}")
#     print("="*80)

#     random.seed(42)

#     print("\n--- 初始化API客户端和Tokenizer ---")
#     try:
#         client = OpenAI(api_key=Config.API_KEY, base_url=Config.API_BASE_URL)
#         tokenizer = tiktoken.get_encoding(Config.TOKENIZER_NAME)
#         print("--- 初始化成功 ---\n")
#     except Exception as e:
#         print(f"错误: 初始化失败: {e}"); exit(1)

#     data_conditions = {"original": Config.BASE_DATA_DIR / "original", "rewrite": Config.BASE_DATA_DIR / "rewrite"}
#     for condition_name, dataset_dir in data_conditions.items():
#         print(f"\n{'='*80}\n开始数据条件: '{condition_name.upper()}'\n{'='*80}")
#         if not dataset_dir.exists():
#             print(f"错误: 数据目录 '{dataset_dir}' 不存在。跳过。")
#             continue
        
#         repo_names = sorted([p.stem for p in dataset_dir.glob("*.jsonl")])
#         repo_iterator = tqdm(repo_names, desc="Repos", position=0)
#         for reponame in repo_iterator:
#             repo_iterator.set_postfix_str(f"Repo: {reponame}")
#             process_repository(reponame, dataset_dir, condition_name, client, tokenizer)
            
#     print("\n--- 所有实验条件均已完成 ---")

# if __name__ == "__main__":
#     main()

import json
import os
from pathlib import Path
import time
import datetime
import random
import re
import argparse
from tqdm import tqdm
from openai import OpenAI, APIError
import tiktoken
import httpx # 导入httpx以支持代理

# ==============================================================================
# 实验配置 (未更改您原有的配置)
# ==============================================================================
class Config:
    # --- 动态API配置 (由命令行参数设置) ---
    API_BASE_URL = None
    API_KEY = None
    MODEL_NAME = None
    MODEL_CONTEXT_WINDOW = None
    
    # --- 静态目录配置 ---
    BASE_DATA_DIR = Path("data_collection_align")
    BM25_RANK_DIR = Path("output_with_bm25_rank")
    CALL_CHAIN_DATA_DIR = Path("output_results")
    REPO_BASE_DIR = Path("python_repos")
    EXPERIMENTS_OUTPUT_DIR = None

    # --- 实验参数 (完全保留您原有的设置) ---
    MAX_COMPLETION_TOKENS = 512
    MAX_ITEMS_PER_REPO = 200
    NUM_SAMPLES_PER_TASK = 5
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REQUEST_DELAY_SECONDS = 1

    # --- Tokenizer (完全保留您原有的设置) ---
    TOKENIZER_NAME = "o200k_base"

    # --- 文件选择参数 (完全保留您原有的设置) ---
    MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL = 20
    NUM_CONFUSION_FILES_TO_SAMPLE = 20
    TOP_N_TO_EXCLUDE_FOR_CONFUSION = 20
    MAX_ORACLE_FILES_TO_CONSIDER = 20

    # --- 派生常量 ---
    MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
    SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 50
    
    # [NEW] 增加网络健壮性参数，不影响实验逻辑
    API_TIMEOUT_SECONDS = 60.0

    PROMPT_TEMPLATE_PARTS = {
"system_message": """Your task is to infer the correct values for the '???' placeholder based on provided Python code.
# Rules
1.  Analyze the provided code and related files to determine the value.
2.  The value MUST be a determined value or a valid Python expression.
3.  **Do not** output any explanations, reasoning, introductions, or any other text outside of the specified output format. Your output must be clean.
# Output format

```output
answer_of_placeholder
```

# Example
Here is an example of a perfect interaction.

## Related files are as follow:
### file: config.py
```python
DEFAULT_TIMEOUT = 50
```

### file: get_timeout.py```python
from config import DEFAULT_TIMEOUT
def get_timeout():
    return DEFAULT_TIMEOUT
```

## Test function is as follows:
### test
```python
def test_timeout():
    from get_timeout import get_timeout
    assert get_timeout() == '???'
```

### Your Correct Output:
```output
50
```
""",
    "user_prompt_template": """## Related files are as follows:
{related_files_content}

## Test function is as follows:
```python
{masked_code_content}
```

Based on all the provided context and the test function, infer the correct value or expression for the "???" placeholder. Output it directly in the required format. Do not output extra information.
"""
    }

# ==============================================================================
# 辅助函数 (无变化)
# ==============================================================================
def count_tokens_custom(text, tok):
    return len(tok.encode(text))

def load_jsonl(file_path):
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def get_related_files_content_budgeted_custom(reponame_arg, related_files_rank_list, current_tokenizer, token_budget_for_related_files):
    content_parts, accumulated_tokens = [], 0
    if token_budget_for_related_files <= 0: return "# 没有为相关文件分配token预算。\n", 0
    for rel_path_str in related_files_rank_list:
        if accumulated_tokens >= token_budget_for_related_files: break
        file_path = Config.REPO_BASE_DIR / reponame_arg / rel_path_str
        file_header_str = f"### file: {rel_path_str}\n```python\n"
        file_footer_str = "\n```\n"
        file_header_tokens = count_tokens_custom(file_header_str, current_tokenizer)
        file_footer_tokens = count_tokens_custom(file_footer_str, current_tokenizer)
        
        file_content_str = ""
        try:
            if file_path.exists() and file_path.is_file():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content_str = f.read()
            else: file_content_str = f"# 文件 {file_path} 未找到"
        except Exception as e: file_content_str = f"# 读取文件错误 {file_path}: {e}"
            
        available_tokens_for_this_file = token_budget_for_related_files - accumulated_tokens
        if file_header_tokens + file_footer_tokens >= available_tokens_for_this_file: break
        
        current_file_str_parts = [file_header_str]
        content_tokens_budget = available_tokens_for_this_file - file_header_tokens - file_footer_tokens
        
        if content_tokens_budget > 0:
            content_token_ids = current_tokenizer.encode(file_content_str)
            if len(content_token_ids) > content_tokens_budget:
                truncated_content_token_ids = content_token_ids[:content_tokens_budget]
                truncated_content_str = current_tokenizer.decode(truncated_content_token_ids)
                current_file_str_parts.append(truncated_content_str)
            else:
                current_file_str_parts.append(file_content_str)
        current_file_str_parts.append(file_footer_str)
        
        full_part = "".join(current_file_str_parts)
        part_tokens = count_tokens_custom(full_part, current_tokenizer)
        
        if accumulated_tokens + part_tokens <= token_budget_for_related_files:
            content_parts.append(full_part)
            accumulated_tokens += part_tokens
        else: break
            
    if not content_parts:
        return "# 没有相关文件内容可以在token限制内被包含。\n", 0
    return "".join(content_parts), accumulated_tokens

def normalize_funcname(name: str) -> str:
    name = re.sub(r'\[.*\]$', '', name)
    if '::' in name:
        name = name.split('::')[-1]
    return name

# ==============================================================================
# 核心实验逻辑
# ==============================================================================
def process_repository(reponame, dataset_dir, condition_name, client, tokenizer):
    dataset_file = dataset_dir / f"{reponame}.jsonl"
    bm25_file = Config.BM25_RANK_DIR / f"{reponame}.jsonl"
    call_chain_file = Config.CALL_CHAIN_DATA_DIR / reponame / "report_functions.jsonl"
    if not dataset_file.exists(): return

    test_data_list = load_jsonl(dataset_file)[:Config.MAX_ITEMS_PER_REPO]
    bm25_lookup = {item['task_id']: item for item in load_jsonl(bm25_file)}
    call_chain_data = load_jsonl(call_chain_file)
    call_chain_lookup = {}
    for item in call_chain_data:
        dirty_func_name = item.get('test_function')
        if not dirty_func_name: continue
        normalized_name = normalize_funcname(dirty_func_name)
        sorted_deps = sorted(item.get('dependencies', []), key=lambda x: x.get('hops', float('inf')))
        call_chain_lookup[normalized_name] = [dep['file'] for dep in sorted_deps]

    tokens_system_message = count_tokens_custom(Config.PROMPT_TEMPLATE_PARTS["system_message"], tokenizer)
    static_user_prompt_template = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content="", masked_code_content="")
    tokens_static_user_prompt = count_tokens_custom(static_user_prompt_template, tokenizer)

    # for experiment_type in ["retrieval", "confusion", "oracle"]:
    for experiment_type in ["oracle"]:
        print(f"    -> 开始实验: {experiment_type}")
        current_preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "predictions" / experiment_type
        current_inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "interactions" / experiment_type
        current_preds_dir.mkdir(parents=True, exist_ok=True)
        current_inters_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = current_preds_dir / f"{reponame}.jsonl"
        interactions_output_file = current_inters_dir / f"{reponame}.jsonl"

        # [NEW] 断点续存: 加载已完成的任务
        completed_tasks = set()
        if predictions_output_file.exists():
            for item in load_jsonl(predictions_output_file):
                task_key = (item.get('task_id'), item.get('sample_id'))
                completed_tasks.add(task_key)
        
        # 仅在需要处理任务时打印消息
        if completed_tasks:
             print(f"      发现 {len(completed_tasks)} 个已完成的任务，将跳过它们。")

        # [MODIFIED] 以追加模式 ('a') 打开文件
        with open(predictions_output_file, 'a', encoding='utf-8') as pred_f, \
             open(interactions_output_file, 'a', encoding='utf-8') as intr_f:
            
            tasks_to_process = []
            for test_case in test_data_list:
                task_id = test_case['task_id']
                related_files_for_experiment = []

                if experiment_type == "retrieval":
                    if task_id in bm25_lookup:
                        files = bm25_lookup[task_id].get('related_files_rank', [])
                        related_files_for_experiment = files[:Config.MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL]
                        random.shuffle(related_files_for_experiment)
                elif experiment_type == "confusion":
                    if task_id in bm25_lookup:
                        files = bm25_lookup[task_id].get('related_files_rank', [])
                        pool = files[Config.TOP_N_TO_EXCLUDE_FOR_CONFUSION:]
                        related_files_for_experiment = random.sample(pool, min(len(pool), Config.NUM_CONFUSION_FILES_TO_SAMPLE))
                elif experiment_type == "oracle":
                    func_name = test_case.get('funcname')
                    if func_name and normalize_funcname(func_name) in call_chain_lookup:
                        files = call_chain_lookup[normalize_funcname(func_name)]
                        related_files_for_experiment = files[:Config.MAX_ORACLE_FILES_TO_CONSIDER]
                
                for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
                    tasks_to_process.append({"test_case": test_case, "related_files": related_files_for_experiment, "sample_idx": sample_idx})
            
            task_iterator = tqdm(tasks_to_process, desc=f"      Tasks", leave=False, position=1)
            for task_item in task_iterator:
                test_case, related_files, sample_idx = task_item["test_case"], task_item["related_files"], task_item["sample_idx"]
                task_id = test_case["task_id"]

                # [NEW] 断点续存: 检查并跳过
                task_key = (task_id, sample_idx)
                if task_key in completed_tasks:
                    continue

                masked_code = test_case['masked_code']
                tokens_test_code = count_tokens_custom(masked_code, tokenizer)
                token_budget_for_dynamic_content = (Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - tokens_system_message - tokens_static_user_prompt - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING)
                token_budget_for_related_files = token_budget_for_dynamic_content - tokens_test_code
                
                related_files_str, _ = get_related_files_content_budgeted_custom(reponame, related_files, tokenizer, token_budget_for_related_files) if token_budget_for_related_files > 0 else ("# 无相关文件上下文。\n", 0)
                
                user_prompt = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content=related_files_str, masked_code_content=masked_code)
                messages = [{"role": "system", "content": Config.PROMPT_TEMPLATE_PARTS["system_message"]}, {"role": "user", "content": user_prompt}]
                
                response_text, finish_reason, usage, error_details = "ERROR: INIT", "error", None, None

                # [FIXED] 更全面的错误处理
                try:
                    response = client.chat.completions.create(
                        model=Config.MODEL_NAME, messages=messages, temperature=Config.TEMPERATURE,
                        top_p=Config.TOP_P, max_tokens=Config.MAX_COMPLETION_TOKENS
                    )
                    response_text = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                    usage = response.usage.model_dump() if response.usage else None
                except APIError as e:
                    print(f"\n      捕获到 API 错误 (Task: {task_id}): {type(e).__name__} - {e}")
                    response_text = f"ERROR: {type(e).__name__} - {str(e)}"
                    status_code = getattr(e, 'status_code', None) # 安全地获取status_code
                    error_details = {"type": type(e).__name__, "message": str(e), "status_code": status_code}
                except Exception as e:
                    print(f"\n      捕获到意外错误 (Task: {task_id}): {type(e).__name__} - {e}")
                    response_text = f"ERROR: SDK/Network Error - {str(e)}"
                    error_details = {"type": type(e).__name__, "message": str(e)}

                intr_f.write(json.dumps({"task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "messages_sent_to_model": messages, "model_used": Config.MODEL_NAME, "related_files_provided_to_prompt": related_files, "api_error_details": error_details}, ensure_ascii=False) + "\n")
                pred_f.write(json.dumps({"task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, "response": response_text.strip(), "finish_reason": finish_reason, "usage": usage}, ensure_ascii=False) + "\n")
                time.sleep(Config.REQUEST_DELAY_SECONDS)

# ==============================================================================
# 主函数 (协调器)
# ==============================================================================
def main():
    CUSTOM_API_KEY = "sk-Eyhl41vT8nBB8xLY77D7C9Bf0198453dA2797221Aa75E486"
    CUSTOM_API_BASE_URL = "https://api.vveai.com/v1/"
    OPENAI_COMPATIBLE_MODEL_NAME = "gpt-4.1-mini"

    
    Config.API_BASE_URL = CUSTOM_API_BASE_URL 
    Config.API_KEY = CUSTOM_API_KEY
    Config.MODEL_NAME = OPENAI_COMPATIBLE_MODEL_NAME
    Config.MODEL_CONTEXT_WINDOW = 10240
    model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.MODEL_NAME)
    Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_output_oracle_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}")
    Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - 100
    
    # 当前实验为gpt-4.1min-10k的oracle实验


    print("="*80)
    print("实验配置初始化完成")
    print(f"  - API Base URL: {Config.API_BASE_URL}")
    print(f"  - Model Name: {Config.MODEL_NAME}")
    print(f"  - 上下文窗口: {Config.MODEL_CONTEXT_WINDOW}")
    print(f"  - 输出目录: {Config.EXPERIMENTS_OUTPUT_DIR}")
    print("="*80)

    random.seed(42)

    print("\n--- 初始化API客户端和Tokenizer ---")
    try:
        http_client = None
        
        # [NEW] 在OpenAI客户端层面设置超时
        client = OpenAI(
            api_key=Config.API_KEY, 
            base_url=Config.API_BASE_URL, 
            http_client=http_client,
            timeout=Config.API_TIMEOUT_SECONDS # 为所有请求设置默认超时
        )
        tokenizer = tiktoken.get_encoding(Config.TOKENIZER_NAME)
        print("--- 初始化成功 ---\n")
    except Exception as e:
        print(f"错误: 初始化失败: {e}"); exit(1)

    data_conditions = {"original": Config.BASE_DATA_DIR / "original", "rewrite": Config.BASE_DATA_DIR / "rewrite"}
    for condition_name, dataset_dir in data_conditions.items():
        print(f"\n{'='*80}\n开始数据条件: '{condition_name.upper()}'\n{'='*80}")
        if not dataset_dir.exists():
            print(f"错误: 数据目录 '{dataset_dir}' 不存在。跳过。")
            continue
        
        repo_names = sorted([p.stem for p in dataset_dir.glob("*.jsonl")])
        repo_iterator = tqdm(repo_names, desc="Repos", position=0)
        for reponame in repo_iterator:
            repo_iterator.set_postfix_str(f"Repo: {reponame}")
            process_repository(reponame, dataset_dir, condition_name, client, tokenizer)
            
    print("\n--- 所有实验条件均已完成 ---")

if __name__ == "__main__":
    main()