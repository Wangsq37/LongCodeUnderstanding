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

# ==============================================================================
# 实验配置
# ==============================================================================
class Config:
    # --- 动态API配置 ---
    API_BASE_URL = None
    API_KEY = None
    MODEL_NAME = None
    MODEL_CONTEXT_WINDOW = 20480
    
    # --- 静态目录配置 ---
    DATA_DIR_BASE = Path("data_collection_align")
    CALL_CHAIN_REPORTS_DIR = Path("output_results")
    REPO_BASE_DIR = Path("python_repos")
    EXPERIMENTS_OUTPUT_DIR = None

    # --- 实验参数 ---
    CALL_CHAIN_RATIOS = [0.25, 0.5]
    MAX_COMPLETION_TOKENS = 1024
    MAX_ITEMS_PER_REPO = 200
    NUM_SAMPLES_PER_TASK = 5
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REQUEST_DELAY_SECONDS = 1

    # --- Tokenizer ---
    TOKENIZER_NAME = "o200k_base"
    
    # --- 派生常量 ---
    MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
    SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 100

    # --- 文件扫描排除项 ---
    EXCLUDED_DIR_NAMES = {'.venv', '.git', 'docs', 'examples', '__pycache__', 'tests', 'test'}
    EXCLUDED_FILE_NAMES = {'__init__.py'}

    PROMPT_TEMPLATE = {
"system_message": """You are an expert Python static analysis assistant. Your task is to analyze a test file and a set of provided code files to identify ONLY the files that are necessary for the execution of the test's call chain.

# Rules
1.  Carefully trace all imports and function calls starting from the test file to identify its dependencies among the provided files.
2.  The test file itself is the starting point but MUST NOT be included in the final output list.
3.  Your output MUST be ONLY a single, clean JSON list of file path strings.
4.  **Do not** output any explanations, reasoning, analysis steps, or any other text. Your response must contain only the final JSON in a markdown block.

# Output Format
```json
["path/to/dependency1.py", "path/to/dependency2.py"]
```

# Example
Here is an example of a perfect interaction.

## Candidate files are as follows:
### file: my_app/calculator.py
```python
def add(a, b):
    return a + b
```
### file: my_app/utils.py
```python
import os
def check_file_exists(path):
    return os.path.exists(path)
```

## The test file is as follows:
### file: tests/test_calculator.py
```python
from my_app.calculator import add
def test_addition():
    assert add(2, 3) == 5
```

## Your Correct Output:
```json
["my_app/calculator.py"]
```
""",

"user_prompt_template": """## The following is a pool of candidate files from the repository.
{files_context}

## The test file is as follows:
### file: {test_file_path}
```python
{test_code}
```
Based on all the provided code, identify the file paths that are part of the test's call chain. Output ONLY the final JSON list as instructed in the rules.
"""
}

# ==============================================================================
# 辅助函数
# ==============================================================================
def count_tokens_custom(text, tok):
    return len(tok.encode(text))

def load_jsonl(file_path):
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: pass
    return data

def get_all_py_files(repo_root):
    py_files = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in Config.EXCLUDED_DIR_NAMES and not d.startswith('.')]
        for file in files:
            if file.endswith('.py') and file not in Config.EXCLUDED_FILE_NAMES:
                full_path = Path(root) / file
                relative_path = full_path.relative_to(repo_root)
                py_files.append(str(relative_path))
    return py_files

def get_files_content_budgeted(files_to_include, reponame, token_budget, tokenizer):
    content_parts, included_files_list = [], []
    accumulated_tokens = 0
    random.shuffle(files_to_include)

    for file_path_str in files_to_include:
        if accumulated_tokens >= token_budget: break
        
        file_path = Config.REPO_BASE_DIR / reponame / file_path_str
        header = f"### file: {file_path_str}\n```python\n"
        footer = "\n```\n"
        header_tokens = count_tokens_custom(header, tokenizer)
        footer_tokens = count_tokens_custom(footer, tokenizer)

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content = f.read()
        except Exception: continue

        available_tokens = token_budget - accumulated_tokens
        if header_tokens + footer_tokens >= available_tokens: continue

        content_budget = available_tokens - header_tokens - footer_tokens
        if content_budget <= 0: continue

        token_ids = tokenizer.encode(file_content)
        if len(token_ids) > content_budget:
            token_ids = token_ids[:content_budget]
        
        final_content = tokenizer.decode(token_ids)
        full_part = header + final_content + footer
        part_tokens = count_tokens_custom(full_part, tokenizer)

        if accumulated_tokens + part_tokens <= token_budget:
            content_parts.append(full_part)
            accumulated_tokens += part_tokens
            included_files_list.append(file_path_str)
    
    return "".join(content_parts), included_files_list

# ==============================================================================
# 核心实验逻辑
# ==============================================================================
def process_repository(reponame, client, tokenizer):
    call_chain_report_file = Config.CALL_CHAIN_REPORTS_DIR / reponame / "report_files.jsonl"
    if not call_chain_report_file.exists(): return
    call_chain_lookup = {item['test_file']: [dep['file'] for dep in item['dependencies']] for item in load_jsonl(call_chain_report_file)}

    original_data_file = Config.DATA_DIR_BASE / "original" / f"{reponame}.jsonl"
    if not original_data_file.exists(): return
    
    seen_test_paths, tasks = set(), []
    for item in load_jsonl(original_data_file):
        test_path = item.get('testpath')
        if test_path and test_path in call_chain_lookup and test_path not in seen_test_paths:
            seen_test_paths.add(test_path)
            tasks.append({'task_id': item['task_id'], 'test_file': test_path})
            if len(tasks) >= Config.MAX_ITEMS_PER_REPO: break
    
    if not tasks: return

    all_repo_py_files = get_all_py_files(Config.REPO_BASE_DIR / reponame)

    for ratio in Config.CALL_CHAIN_RATIOS:
        ratio_str = f"{int(ratio*100)}_percent_positive"
        print(f"    -> 开始实验: {ratio_str}")

        preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / "predictions" / ratio_str
        inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / "interactions" / ratio_str
        preds_dir.mkdir(parents=True, exist_ok=True)
        inters_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = preds_dir / f"{reponame}.jsonl"
        interactions_output_file = inters_dir / f"{reponame}.jsonl"

        with open(predictions_output_file, 'w', encoding='utf-8') as pred_f, \
             open(interactions_output_file, 'w', encoding='utf-8') as intr_f:

            tasks_to_process = []
            for task in tasks:
                positive_pool = [f for f in call_chain_lookup[task['test_file']] if f != task['test_file']]
                negative_pool = [f for f in all_repo_py_files if f not in call_chain_lookup[task['test_file']]]
                for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
                    tasks_to_process.append({"task_info": task, "positive_pool": positive_pool, "negative_pool": negative_pool, "sample_idx": sample_idx})
            
            task_iterator = tqdm(tasks_to_process, desc=f"      Tasks", leave=False, position=1)
            for task_item in task_iterator:
                test_file_path = task_item['task_info']['test_file']
                
                try:
                    with open(Config.REPO_BASE_DIR / reponame / test_file_path, 'r', encoding='utf-8') as f: test_code = f.read()
                except Exception: continue

                static_prompt_tokens = count_tokens_custom(Config.PROMPT_TEMPLATE['system_message'], tokenizer) + count_tokens_custom(Config.PROMPT_TEMPLATE['user_prompt_template'].format(test_file_path="", test_code=test_code, files_context=""), tokenizer)
                files_total_budget = Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - static_prompt_tokens
                positive_files_budget = int(files_total_budget * ratio)
                negative_files_budget = files_total_budget - positive_files_budget

                pos_content, included_pos = get_files_content_budgeted(task_item['positive_pool'], reponame, positive_files_budget, tokenizer)
                neg_content, included_neg = get_files_content_budgeted(task_item['negative_pool'], reponame, negative_files_budget, tokenizer)
                
                context_parts = [pos_content, neg_content]
                random.shuffle(context_parts)
                files_context_str = "".join(context_parts)
                
                user_prompt = Config.PROMPT_TEMPLATE['user_prompt_template'].format(test_file_path=test_file_path, test_code=test_code, files_context=files_context_str)
                messages = [{"role": "system", "content": Config.PROMPT_TEMPLATE['system_message']}, {"role": "user", "content": user_prompt}]
                
                response_text, finish_reason, usage, error_details = "ERROR: INIT", "error", None, None
                try:
                    response = client.chat.completions.create(model=Config.MODEL_NAME, messages=messages, temperature=Config.TEMPERATURE, top_p=Config.TOP_P, max_tokens=Config.MAX_COMPLETION_TOKENS)
                    response_text, finish_reason = response.choices[0].message.content, response.choices[0].finish_reason
                    usage = response.usage.model_dump() if response.usage else None
                except APIError as e:
                    response_text, error_details = f"ERROR: APIError - {e.message}", {"type": type(e).__name__, "message": str(e), "status_code": e.status_code}
                except Exception as e:
                    response_text, error_details = f"ERROR: SDK/Network Error - {type(e).__name__}", {"type": type(e).__name__, "message": str(e)}

                task_info = task_item['task_info']
                intr_f.write(json.dumps({"task_id": task_info['task_id'], "test_file": task_info['test_file'], "sample_id": task_item['sample_idx'], "ratio": ratio, "messages": messages, "files_in_prompt": included_pos + included_neg, "ground_truth_chain": task_item['positive_pool'], "error": error_details}, ensure_ascii=False) + "\n")
                pred_f.write(json.dumps({"task_id": task_info['task_id'], "test_file": task_info['test_file'], "sample_id": task_item['sample_idx'], "ratio": ratio, "response": response_text.strip(), "finish_reason": finish_reason, "usage": usage}, ensure_ascii=False) + "\n")
                time.sleep(Config.REQUEST_DELAY_SECONDS)

# ==============================================================================
# 主函数 (协调器)
# ==============================================================================
def main():
    # parser = argparse.ArgumentParser(description="运行调用链预测实验 (API版本)。")
    # parser.add_argument("--api_base_url", type=str, required=True, help="OpenAI兼容API的基础URL。")
    # parser.add_argument("--api_key", type=str, required=True, help="API密钥。")
    # parser.add_argument("--model_name", type=str, required=True, help="要使用的模型名称。")
    # parser.add_argument("--context_window", type=int, default=20480, help="模型的上下文窗口大小。")
    # parser.add_argument("--ratios", nargs='+', type=float, default=[0.25, 0.5], help="调用链文件token预算比例。")
    # args = parser.parse_args()

    # Config.API_BASE_URL = args.api_base_url
    # Config.API_KEY = args.api_key
    # Config.MODEL_NAME = args.model_name
    CUSTOM_API_KEY = "sk-Eyhl41vT8nBB8xLY77D7C9Bf0198453dA2797221Aa75E486"
    CUSTOM_API_BASE_URL = "https://api.vveai.com/v1/"
    OPENAI_COMPATIBLE_MODEL_NAME = "gpt-4.1-mini"

    
    Config.API_BASE_URL = CUSTOM_API_BASE_URL 
    Config.API_KEY = CUSTOM_API_KEY
    Config.MODEL_NAME = OPENAI_COMPATIBLE_MODEL_NAME
    Config.MODEL_CONTEXT_WINDOW = 20480
    Config.CALL_CHAIN_RATIOS = [0.25, 0.5]
    model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.MODEL_NAME)
    Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_callchain_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}")
    Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING

    print("="*80)
    print("实验配置初始化完成")
    print(f"  - API Base URL: {Config.API_BASE_URL}")
    print(f"  - Model Name: {Config.MODEL_NAME}")
    print(f"  - 上下文窗口: {Config.MODEL_CONTEXT_WINDOW}")
    print(f"  - 调用链比例: {Config.CALL_CHAIN_RATIOS}")
    print(f"  - 输出目录: {Config.EXPERIMENTS_OUTPUT_DIR}")
    print("="*80)

    random.seed(42)

    print("\n--- 初始化API客户端和Tokenizer ---")
    try:
        client = OpenAI(api_key=Config.API_KEY, base_url=Config.API_BASE_URL)
        tokenizer = tiktoken.get_encoding(Config.TOKENIZER_NAME)
        print("--- 初始化成功 ---\n")
    except Exception as e:
        print(f"错误: 初始化失败: {e}"); exit(1)

    repo_names = sorted([p.stem for p in (Config.CALL_CHAIN_REPORTS_DIR).iterdir() if p.is_dir()])
    repo_iterator = tqdm(repo_names, desc="Repos", position=0)
    for reponame in repo_iterator:
        repo_iterator.set_postfix_str(f"Repo: {reponame}")
        if not (Config.REPO_BASE_DIR / reponame).exists(): continue
        process_repository(reponame, client, tokenizer)
            
    print("\n--- 所有实验已完成 ---")

if __name__ == "__main__":
    main()