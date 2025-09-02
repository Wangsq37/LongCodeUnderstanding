# -*- coding: utf-8 -*-

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from pathlib import Path

class DataQualityReporter:
    """
    数据质量报告生成器，用于分析数据质量过滤的效果
    """
    
    def __init__(self):
        self.stats = {
            'total_assertions': 0,
            'quality_assertions': 0,
            'filtered_assertions': 0,
            'repos_processed': 0,
            'quality_by_repo': defaultdict(dict),
            'complexity_distribution': Counter(),
            'filter_reasons': Counter()
        }
    
    def analyze_data_file(self, file_path: str) -> Dict[str, any]:
        """
        分析单个数据文件的质量统计
        """
        if not os.path.exists(file_path):
            return {}
        
        repo_stats = {
            'total_assertions': 0,
            'quality_assertions': 0,
            'filtered_assertions': 0,
            'complexity_scores': [],
            'filter_reasons': Counter()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        repo_stats['total_assertions'] += 1
                        
                        # 如果有质量分析信息，进行统计
                        if 'quality_analysis' in data:
                            analysis = data['quality_analysis']
                            complexity = analysis.get('complexity_score', 0)
                            repo_stats['complexity_scores'].append(complexity)
                            
                            if analysis.get('is_quality', False):
                                repo_stats['quality_assertions'] += 1
                            else:
                                repo_stats['filtered_assertions'] += 1
                                reason = analysis.get('reason', 'Unknown')
                                repo_stats['filter_reasons'][reason] += 1
                        else:
                            # 如果没有质量分析信息，假设是通过了过滤
                            repo_stats['quality_assertions'] += 1
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"警告: 无法分析文件 {file_path}: {e}", file=sys.stderr)
            return {}
        
        return repo_stats
    
    def generate_quality_report(self, data_dir: str, output_file: str = None):
        """
        生成数据质量报告
        """
        data_path = Path(data_dir)
        if not data_path.exists() or not data_path.is_dir():
            print(f"错误: 数据目录 '{data_dir}' 不存在", file=sys.stderr)
            return
        
        # 分析所有数据文件
        jsonl_files = list(data_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"警告: 在目录 '{data_dir}' 中未找到任何 .jsonl 文件", file=sys.stderr)
            return
        
        print(f"正在分析 {len(jsonl_files)} 个数据文件...")
        
        for file_path in jsonl_files:
            reponame = file_path.stem
            repo_stats = self.analyze_data_file(str(file_path))
            
            if repo_stats:
                self.stats['repos_processed'] += 1
                self.stats['total_assertions'] += repo_stats['total_assertions']
                self.stats['quality_assertions'] += repo_stats['quality_assertions']
                self.stats['filtered_assertions'] += repo_stats['filtered_assertions']
                
                # 记录每个仓库的统计
                self.stats['quality_by_repo'][reponame] = repo_stats
                
                # 记录复杂度分布
                for score in repo_stats['complexity_scores']:
                    self.stats['complexity_distribution'][score] += 1
                
                # 记录过滤原因
                for reason, count in repo_stats['filter_reasons'].items():
                    self.stats['filter_reasons'][reason] += count
        
        # 生成报告
        self._print_report()
        
        # 保存详细报告到文件
        if output_file:
            self._save_detailed_report(output_file)
    
    def _print_report(self):
        """
        打印质量报告到控制台
        """
        print("\n" + "=" * 80)
        print("数据质量分析报告")
        print("=" * 80)
        
        total = self.stats['total_assertions']
        quality = self.stats['quality_assertions']
        filtered = self.stats['filtered_assertions']
        
        print(f"\n总体统计:")
        print(f"  - 处理的仓库数量: {self.stats['repos_processed']}")
        print(f"  - 总断言数量: {total}")
        print(f"  - 高质量断言: {quality} ({quality/total*100:.1f}%)")
        print(f"  - 被过滤断言: {filtered} ({filtered/total*100:.1f}%)")
        
        if self.stats['complexity_distribution']:
            print(f"\n复杂度分布:")
            for score in sorted(self.stats['complexity_distribution'].keys()):
                count = self.stats['complexity_distribution'][score]
                print(f"  - 复杂度 {score}: {count} 个断言")
        
        if self.stats['filter_reasons']:
            print(f"\n过滤原因统计:")
            for reason, count in self.stats['filter_reasons'].most_common():
                print(f"  - {reason}: {count} 个断言")
        
        print(f"\n各仓库质量统计:")
        for reponame, repo_stats in sorted(self.stats['quality_by_repo'].items()):
            repo_total = repo_stats['total_assertions']
            repo_quality = repo_stats['quality_assertions']
            quality_rate = repo_quality / repo_total * 100 if repo_total > 0 else 0
            print(f"  - {reponame}: {repo_quality}/{repo_total} ({quality_rate:.1f}%)")
        
        print("=" * 80)
    
    def _save_detailed_report(self, output_file: str):
        """
        保存详细报告到文件
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            print(f"\n详细报告已保存到: {output_file}")
        except Exception as e:
            print(f"警告: 无法保存详细报告到 {output_file}: {e}", file=sys.stderr)

def main():
    """
    主函数，用于命令行调用
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="数据质量分析报告生成器")
    parser.add_argument('data_dir', help="包含 .jsonl 数据文件的目录")
    parser.add_argument('--output', '-o', help="输出详细报告的JSON文件路径")
    
    args = parser.parse_args()
    
    reporter = DataQualityReporter()
    reporter.generate_quality_report(args.data_dir, args.output)

if __name__ == '__main__':
    main()
