# -*- coding: utf-8 -*-

"""
conftest.py调用链跟踪禁用工具
用于在重写阶段临时禁用调用链跟踪功能，而不影响其他conftest.py配置
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def disable_call_chain_tracking_in_conftest(conftest_path: Path) -> Tuple[bool, str]:
    """
    在conftest.py中临时禁用调用链跟踪功能
    
    Args:
        conftest_path: conftest.py文件路径
        
    Returns:
        (是否修改成功, 原始内容)
    """
    if not conftest_path.exists():
        return False, ""
    
    try:
        content = conftest_path.read_text(encoding='utf-8')
        
        # 检查是否包含调用链跟踪代码
        if 'hunter.trace' not in content and 'FileTracer' not in content:
            return False, content  # 没有调用链跟踪代码，无需修改
        
        # 临时禁用调用链跟踪
        modified_content = _disable_tracking_functions(content)
        
        # 写回文件
        conftest_path.write_text(modified_content, encoding='utf-8')
        
        return True, content
        
    except Exception as e:
        print(f"修改conftest.py失败: {e}")
        return False, ""

def restore_conftest_content(conftest_path: Path, original_content: str) -> bool:
    """
    恢复conftest.py的原始内容
    
    Args:
        conftest_path: conftest.py文件路径
        original_content: 原始内容
        
    Returns:
        是否恢复成功
    """
    try:
        conftest_path.write_text(original_content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"恢复conftest.py失败: {e}")
        return False

def _disable_tracking_functions(content: str) -> str:
    """
    禁用调用链跟踪函数
    """
    lines = content.split('\n')
    modified_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是pytest_runtest_protocol函数
        if 'def pytest_runtest_protocol(' in line:
            modified_lines.append(line)
            modified_lines.append('    # Call chain tracking disabled during rewrite phase')
            modified_lines.append('    yield')
            modified_lines.append('    return')
            
            # 跳过函数体直到函数结束
            i += 1
            indent_level = len(line) - len(line.lstrip())
            while i < len(lines):
                current_line = lines[i]
                if current_line.strip() and len(current_line) - len(current_line.lstrip()) <= indent_level:
                    break
                i += 1
            continue
        
        # 检查是否是pytest_sessionfinish函数
        elif 'def pytest_sessionfinish(' in line:
            modified_lines.append(line)
            modified_lines.append('    # Call chain tracking disabled during rewrite phase')
            modified_lines.append('    return')
            
            # 跳过函数体直到函数结束
            i += 1
            indent_level = len(line) - len(line.lstrip())
            while i < len(lines):
                current_line = lines[i]
                if current_line.strip() and len(current_line) - len(current_line.lstrip()) <= indent_level:
                    break
                i += 1
            continue
        
        else:
            modified_lines.append(line)
            i += 1
    
    return '\n'.join(modified_lines)

def find_and_disable_conftest_files(repo_path: Path) -> List[Tuple[Path, str]]:
    """
    查找并禁用仓库中所有conftest.py文件的调用链跟踪
    
    Args:
        repo_path: 仓库根目录
        
    Returns:
        修改的文件列表，每个元素为(文件路径, 原始内容)
    """
    modified_files = []
    
    for conftest_file in repo_path.rglob("conftest.py"):
        success, original_content = disable_call_chain_tracking_in_conftest(conftest_file)
        if success:
            modified_files.append((conftest_file, original_content))
            print(f"已禁用调用链跟踪: {conftest_file}")
    
    return modified_files

def restore_all_conftest_files(modified_files: List[Tuple[Path, str]]):
    """
    恢复所有修改的conftest.py文件
    
    Args:
        modified_files: 修改的文件列表
    """
    for conftest_file, original_content in modified_files:
        if restore_conftest_content(conftest_file, original_content):
            print(f"已恢复: {conftest_file}")
        else:
            print(f"恢复失败: {conftest_file}")

# 使用示例
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python conftest_disable_tracking.py <repo_path>")
        sys.exit(1)
    
    repo_path = Path(sys.argv[1])
    if not repo_path.exists():
        print(f"仓库路径不存在: {repo_path}")
        sys.exit(1)
    
    print(f"在 {repo_path} 中查找并禁用conftest.py调用链跟踪...")
    modified_files = find_and_disable_conftest_files(repo_path)
    
    if modified_files:
        print(f"已修改 {len(modified_files)} 个conftest.py文件")
        input("按回车键恢复文件...")
        restore_all_conftest_files(modified_files)
    else:
        print("未找到需要修改的conftest.py文件")
