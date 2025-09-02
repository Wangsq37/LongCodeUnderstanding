import json
import os
from pathlib import Path
import sys

# 您需要先安装这个库: pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("错误: 依赖库 'rank-bm25' 未安装。请运行 'pip install rank-bm25' 进行安装。", file=sys.stderr)
    sys.exit(1)

# 用于缓存每个仓库的语料库，避免重复构建
CORPUS_CACHE = {}

# 定义要全局排除的目录名称集合，方便管理
EXCLUDED_DIR_NAMES = {
    # 用户指定的
    'venv', 'node_modules', 'build', 'dist', 'site-packages', 'tests', 'test', 'testing',
    # 常见的其他无关目录
    '.venv', '.git', '.hg', 'docs', 'examples', 'samples', 'scripts'
}

def create_corpus_for_repo(repo_path: Path):
    """
    为单个仓库构建语料库，排除测试和无关文件/目录。
    返回文件路径列表和分词后的内容列表。
    """
    repo_key = str(repo_path)
    if repo_key in CORPUS_CACHE:
        return CORPUS_CACHE[repo_key]

    print(f"  - 正在为仓库 '{repo_path.name}' 构建语料库...")
    doc_paths = []
    tokenized_corpus = []

    all_py_files = list(repo_path.rglob("*.py"))

    for file_path in all_py_files:
        path_parts = {p.lower() for p in file_path.parts}

        if not EXCLUDED_DIR_NAMES.isdisjoint(path_parts):
            continue
        
        if file_path.name in ("__init__.py", "setup.py", "conftest.py"):
            continue
        
        try:
            content = file_path.read_text(encoding="utf-8")
            tokenized_content = content.split()
            
            if tokenized_content:
                doc_paths.append(file_path.relative_to(repo_path).as_posix())
                tokenized_corpus.append(tokenized_content)
        except Exception:
            continue
    
    print(f"  - 语料库构建完成，包含 {len(doc_paths)} 个源文件。")
    result = (doc_paths, tokenized_corpus)
    CORPUS_CACHE[repo_key] = result
    return result

# --- 主要修改区域 ---
def analyze_with_bm25(input_jsonl, output_jsonl, repos_base_dir):
    """
    使用BM25处理单个jsonl文件，并缓存重复的查询。
    """
    reponame = Path(input_jsonl).stem
    repo_root = Path(repos_base_dir) / "python_repos" / reponame

    if not repo_root.is_dir():
        print(f"错误: 找不到仓库 '{reponame}' 对应的目录 '{repo_root}'。", file=sys.stderr)
        return

    corpus_paths, tokenized_corpus = create_corpus_for_repo(repo_root)
    
    if not corpus_paths:
        print(f"警告: 仓库 '{reponame}' 的语料库为空，将跳过处理。", file=sys.stderr)
        return
        
    bm25 = BM25Okapi(tokenized_corpus)

    # ✨ 新增：用于缓存每个 testpath 的 BM25 检索结果
    # 键是 testpath，值是检索到的相关文件列表
    query_results_cache = {}

    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        print(f"  - 开始处理 {reponame} 中的条目...")
        for line in infile:
            try:
                data = json.loads(line)
                testpath = data.get("testpath")

                if not testpath:
                    continue

                # ✨ 核心逻辑：检查缓存
                if testpath in query_results_cache:
                    # 如果此测试文件的结果已在缓存中，直接使用
                    ranked_files = query_results_cache[testpath]
                else:
                    # 如果是首次遇到此测试文件，则执行检索
                    # print(f"    - 正在为新文件 '{testpath}' 执行BM25检索...") # (可选) 打印调试信息
                    query_test_path = repo_root / testpath

                    if not query_test_path.is_file():
                        ranked_files = []
                    else:
                        query_content = query_test_path.read_text(encoding="utf-8")
                        tokenized_query = query_content.split()
                        ranked_files = bm25.get_top_n(tokenized_query, corpus_paths, n=len(corpus_paths))

                    # ✨ 将新结果存入缓存，以备后续使用
                    query_results_cache[testpath] = ranked_files

                # 构建并写入输出数据
                output_data = {
                    "task_id": data.get("task_id"),
                    "reponame": data.get("reponame"),
                    "testpath": data.get("testpath"),
                    "testname": data.get("testname"),
                    "funcname": data.get("funcname"),
                    "related_files_rank": ranked_files
                }
                
                outfile.write(json.dumps(output_data) + '\n')

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                # 打印错误信息，方便调试，但程序继续运行
                print(f"    - 警告: 处理行时出错 (错误: {e})，已跳过。行内容: {line.strip()}", file=sys.stderr)
                continue

def main():
    """
    批量处理主逻辑。
    """
    # ✨ 建议将目录配置为绝对路径或相对于脚本的位置，以增加稳健性
    script_dir = Path(__file__).parent
    REPOS_BASE_DIR = script_dir
    SOURCE_DATA_DIR = script_dir / 'data_collection_align' / 'original'
    OUTPUT_DIR = script_dir / 'output_with_bm25_rank'
    
    if not SOURCE_DATA_DIR.is_dir():
        print(f"错误: 输入目录 '{SOURCE_DATA_DIR}' 不存在。", file=sys.stderr)
        return
        
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    files_to_process = [f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith('.jsonl')]
    
    print(f"找到 {len(files_to_process)} 个文件进行处理，输出到 '{OUTPUT_DIR}' 目录...")

    for filename in files_to_process:
        input_path = SOURCE_DATA_DIR / filename
        output_path = OUTPUT_DIR / filename
        
        print(f"\n[+] 正在处理: {filename}")
        analyze_with_bm25(str(input_path), str(output_path), str(REPOS_BASE_DIR))
        print(f"[✓] 处理完成: 结果已保存到 {output_path}")
        
    print("\n所有文件处理完毕！")

if __name__ == '__main__':
    main()
