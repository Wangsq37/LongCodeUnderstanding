# -*- coding: utf-8 -*-

import ast
import astunparse
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

class DataQualityFilter:
    """
    数据质量过滤器，用于评估和过滤测试断言的质量
    """
    
    def __init__(self):
        # 定义简单模式，这些模式通常表示低质量数据
        self.simple_patterns = [
            # 简单的字符串字面量比较
            r"assert\s+['\"][^'\"]*['\"]\s*==\s*\w+",
            # 简单的数字字面量比较
            r"assert\s+\d+\s*==\s*\w+",
            # 简单的布尔值字面量比较
            r"assert\s+(True|False)\s*==\s*\w+",
            # 简单的None字面量比较
            r"assert\s+None\s*==\s*\w+",
            # 简单的变量名比较（两个都是简单变量名）
            r"assert\s+\b[a-zA-Z_]\w*\b\s*==\s*\b[a-zA-Z_]\w*\b",
            # 简单的repr比较（如 repr(Is(5)) == '5'）
            r"assert\s+repr\([^)]+\)\s*==\s*['\"][^'\"]*['\"]",
            # 简单的str比较（如 str(5) == '5'）
            r"assert\s+str\([^)]+\)\s*==\s*['\"][^'\"]*['\"]",
        ]
        
        # 定义高质量模式，这些模式通常表示有意义的数据
        self.quality_patterns = [
            # 函数调用在左边
            r"assert\s+\w+\([^)]*\)\s*==\s*",
            # 方法调用在左边
            r"assert\s+\w+\.\w+\([^)]*\)\s*==\s*",
            # 属性访问在左边
            r"assert\s+\w+\.\w+\s*==\s*",
            # 索引访问在左边
            r"assert\s+\w+\[[^\]]*\]\s*==\s*",
            # 切片操作在左边
            r"assert\s+\w+\[[^\]]*:[^\]]*\]\s*==\s*",
            # 数学运算在左边
            r"assert\s+[^=]*[\+\-\*/%][^=]*\s*==\s*",
            # 逻辑运算在左边
            r"assert\s+[^=]*(and|or|not)[^=]*\s*==\s*",
            # 比较运算在左边
            r"assert\s+[^=]*[<>!=][^=]*\s*==\s*",
        ]
    
    def analyze_assertion_complexity(self, assert_node: ast.Assert) -> Dict[str, any]:
        """
        分析断言语句的复杂度
        """
        if not isinstance(assert_node.test, ast.Compare):
            return {"complexity_score": 0, "is_quality": False, "reason": "Not a comparison"}
        
        if not assert_node.test.ops or not isinstance(assert_node.test.ops[0], ast.Eq):
            return {"complexity_score": 0, "is_quality": False, "reason": "Not an equality comparison"}
        
        left_side = assert_node.test.left
        right_side = assert_node.test.comparators[0]
        
        # 分析左边（被测试的表达式）
        left_complexity = self._analyze_expression_complexity(left_side)
        
        # 分析右边（期望值）
        right_complexity = self._analyze_expression_complexity(right_side)
        
        # 计算总体复杂度分数
        total_complexity = left_complexity + right_complexity
        
        # 判断是否为高质量数据
        is_quality = self._is_quality_assertion(left_side, right_side, total_complexity)
        
        return {
            "complexity_score": total_complexity,
            "left_complexity": left_complexity,
            "right_complexity": right_complexity,
            "is_quality": is_quality,
            "reason": self._get_quality_reason(left_side, right_side, total_complexity)
        }
    
    def _analyze_expression_complexity(self, node: ast.AST) -> int:
        """
        分析表达式的复杂度
        """
        if node is None:
            return 0
        
        complexity = 0
        
        # 基础节点类型
        if isinstance(node, (ast.Name, ast.Constant, ast.Num, ast.Str)):
            complexity = 1
        elif isinstance(node, ast.Attribute):
            complexity = 2
        elif isinstance(node, ast.Call):
            complexity = 3
            # 递归分析参数
            for arg in node.args:
                complexity += self._analyze_expression_complexity(arg)
        elif isinstance(node, ast.Subscript):
            complexity = 3
            complexity += self._analyze_expression_complexity(node.value)
            complexity += self._analyze_expression_complexity(node.slice)
        elif isinstance(node, ast.BinOp):
            complexity = 2
            complexity += self._analyze_expression_complexity(node.left)
            complexity += self._analyze_expression_complexity(node.right)
        elif isinstance(node, ast.UnaryOp):
            complexity = 2
            complexity += self._analyze_expression_complexity(node.operand)
        elif isinstance(node, ast.Compare):
            complexity = 2
            complexity += self._analyze_expression_complexity(node.left)
            for comparator in node.comparators:
                complexity += self._analyze_expression_complexity(comparator)
        elif isinstance(node, ast.BoolOp):
            complexity = 2
            for value in node.values:
                complexity += self._analyze_expression_complexity(value)
        elif isinstance(node, ast.List):
            complexity = 2
            for elt in node.elts:
                complexity += self._analyze_expression_complexity(elt)
        elif isinstance(node, ast.Tuple):
            complexity = 2
            for elt in node.elts:
                complexity += self._analyze_expression_complexity(elt)
        elif isinstance(node, ast.Dict):
            complexity = 3
            for key, value in zip(node.keys, node.values):
                complexity += self._analyze_expression_complexity(key)
                complexity += self._analyze_expression_complexity(value)
        
        return complexity
    
    def _is_quality_assertion(self, left_side: ast.AST, right_side: ast.AST, total_complexity: int) -> bool:
        """
        判断断言是否为高质量数据
        主要关注断言语句的模式，而不是复杂度
        """
        # 1. 避免明显的简单模式匹配
        assert_str = astunparse.unparse(ast.Assert(ast.Compare(left_side, [ast.Eq()], [right_side])))
        if self._matches_simple_patterns(assert_str):
            return False
        
        # 2. 检查是否为有意义的测试模式
        # 高质量的模式包括：
        # - 函数调用结果比较
        # - 方法调用结果比较  
        # - 属性访问比较
        # - 索引/切片操作比较
        # - 数学运算结果比较
        # - 逻辑运算结果比较
        
        # 3. 检查左边是否包含有意义的操作
        left_has_meaningful_operation = self._has_meaningful_operation(left_side)
        
        # 4. 检查右边是否不太简单（避免直接猜测）
        right_is_not_too_simple = self._is_not_too_simple(right_side)
        
        return left_has_meaningful_operation and right_is_not_too_simple
    
    def _has_meaningful_operation(self, node: ast.AST) -> bool:
        """
        检查表达式是否包含有意义的操作
        这些操作通常表示测试在验证某种计算或处理的结果
        """
        # 直接检查当前节点
        if isinstance(node, ast.Call):
            return True  # 函数调用
        elif isinstance(node, ast.Attribute):
            return True  # 方法调用或属性访问
        elif isinstance(node, ast.Subscript):
            return True  # 索引或切片操作
        elif isinstance(node, ast.BinOp):
            return True  # 数学运算
        elif isinstance(node, ast.UnaryOp):
            return True  # 一元运算
        elif isinstance(node, ast.Compare):
            return True  # 比较运算
        elif isinstance(node, ast.BoolOp):
            return True  # 逻辑运算
        elif isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            return True  # 容器操作
        
        # 递归检查子节点
        for child in ast.walk(node):
            if isinstance(child, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
                return True
        
        return False
    
    def _is_not_too_simple(self, node: ast.AST) -> bool:
        """
        检查右边表达式是否不太简单
        主要过滤掉过于明显的字面量
        """
        if isinstance(node, (ast.Constant, ast.Num, ast.Str)):
            # 检查是否为简单的字面量
            if isinstance(node, ast.Constant):
                value = node.value
            elif isinstance(node, ast.Num):
                value = node.n
            elif isinstance(node, ast.Str):
                value = node.s
            else:
                value = None
            
            # 过滤掉过于简单的字面量
            if isinstance(value, (int, float)):
                # 数字：过滤掉0, 1, -1等过于简单的数字
                if value in [0, 1, -1, 2, -2]:
                    return False
            elif isinstance(value, str):
                # 字符串：过滤掉空字符串和过短的字符串
                if len(value) <= 1:
                    return False
                # 过滤掉明显的简单字符串
                if value.lower() in ['true', 'false', 'none', 'null', 'yes', 'no']:
                    return False
        
        elif isinstance(node, ast.Name):
            # 变量名：过滤掉过于简单的变量名
            if node.id in ['True', 'False', 'None', 'null', 'yes', 'no']:
                return False
        
        return True
    
    def _matches_simple_patterns(self, assert_str: str) -> bool:
        """
        检查断言字符串是否匹配简单模式
        """
        for pattern in self.simple_patterns:
            if re.search(pattern, assert_str, re.IGNORECASE):
                return True
        return False
    
    def _get_quality_reason(self, left_side: ast.AST, right_side: ast.AST, total_complexity: int) -> str:
        """
        获取质量评估的原因
        """
        assert_str = astunparse.unparse(ast.Assert(ast.Compare(left_side, [ast.Eq()], [right_side])))
        if self._matches_simple_patterns(assert_str):
            return "Matches simple pattern"
        
        if not self._has_meaningful_operation(left_side):
            return "Left side lacks meaningful operations"
        
        if not self._is_not_too_simple(right_side):
            return "Right side too simple"
        
        return "High quality assertion"
    
    def filter_test_function(self, func_node: ast.FunctionDef) -> List[Dict[str, any]]:
        """
        过滤测试函数中的断言，返回高质量的断言数据
        """
        quality_asserts = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                analysis = self.analyze_assertion_complexity(node)
                if analysis["is_quality"]:
                    quality_asserts.append({
                        "assert_node": node,
                        "analysis": analysis
                    })
        
        return quality_asserts

def create_quality_filter():
    """
    创建数据质量过滤器实例
    """
    return DataQualityFilter()
