# file path: runnable_agent_batch/utils.py

import os
import subprocess
from pathlib import Path
import tiktoken

def get_project_structure(repo_path: Path, max_depth=3, max_files_per_dir=5, encoding: tiktoken.Encoding = None) -> str:
    """
    Get a summarized project structure, limiting depth and file count for clarity.
    """
    repo_path = repo_path.resolve()
    exclude_patterns = {
        '.git', '.svn', '.hg', '.DS_Store', 'Thumbs.db', '.*~', '*.tmp', '*.temp',
        '__pycache__', '.pytest_cache', '*.egg-info',
        'node_modules', '.npm', '.yarn',
        '*.class', 'target',
        '*.o', '*.obj', '*.exe', '*.dll', '*.so', '*.dylib',
        'obj', '*.pdb',
        'build', 'dist', 'out',
        '.vscode', '.idea', '*.swp', '*.swo'
    }

    structure = []
    
    def _is_excluded(path: Path) -> bool:
        """Check if a path should be excluded."""
        return any(path.match(p) or p in path.parts for p in exclude_patterns)

    def build_structure(current_path: Path, prefix: str = '', depth: int = 0):
        if depth > max_depth or _is_excluded(current_path):
            return

        # 获取目录内容并分类
        try:
            # 使用 try-except 块处理可能的 PermissionError
            contents = sorted(list(current_path.iterdir()), key=lambda p: (p.is_file(), p.name.lower()))
        except (PermissionError, FileNotFoundError):
            return

        # 过滤掉需要排除的内容
        contents = [p for p in contents if not _is_excluded(p)]
        
        # 进一步分离目录和文件
        dirs = [p for p in contents if p.is_dir()]
        files = [p for p in contents if p.is_file()]

        # 合并目录和文件，目录优先
        display_items = dirs + files
        
        # 准备输出
        pointers = ['├── '] * (len(display_items) - 1) + ['└── ']
        
        for i, path in enumerate(display_items):
            structure.append(f"{prefix}{pointers[i]}{path.name}{'/' if path.is_dir() else ''}")
            
            if path.is_dir():
                extension = '│   ' if i < len(display_items) - 1 else '    '
                build_structure(path, prefix + extension, depth + 1)
        
        # 如果文件太多，进行截断提示
        if len(files) > max_files_per_dir:
            structure.append(f"{prefix}    └── ... ({len(files) - max_files_per_dir} more files)")


    structure.append(f"{repo_path.name}/")
    build_structure(repo_path)
    
    # 使用 tiktoken 来确保最终输出不会超长，尽管新的结构函数已经很紧凑了
    return truncate_content("\n".join(structure), 4000, encoding)


def truncate_content(content: str, max_tokens: int, encoding: tiktoken.Encoding = None) -> str:
    """Truncate content to fit within token limit"""
    if not encoding:
        return content # 如果没有提供编码器，则返回原始内容
        
    token_list = encoding.encode(content)
    
    if len(token_list) <= max_tokens:
        return content
    
    original_length = len(token_list)
    token_list = token_list[:max_tokens]
    
    # 解码截断后的内容
    truncated_text = encoding.decode(token_list)
    
    # 计算原始内容和截断后内容的行数
    original_lines = len(content.splitlines())
    truncated_lines = len(truncated_text.splitlines())
    omitted_lines = original_lines - truncated_lines
    
    # 在最后加上省略提示
    if omitted_lines > 0:
        return truncated_text + f'\n... ({omitted_lines} lines omitted)'
    
    # 如果没有省略行，但 token 截断了最后一行，则添加省略号
    if len(content) > len(truncated_text):
         return truncated_text + '\n... (truncated)'

    return truncated_text

def read_file_content(file_path: Path, max_tokens: int = 2000, encoding: tiktoken.Encoding = None) -> str:
    """Read and return file content with token limit"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip():
                return truncate_content(content, max_tokens, encoding)
            return ""
    except (UnicodeDecodeError, PermissionError, FileNotFoundError):
        return ""