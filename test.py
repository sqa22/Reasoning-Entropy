import json
import os
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy import latex2sympy
import re
from math import isclose
import sympy as sp
from symeval import EvaluatorMathBatch

evaluator = EvaluatorMathBatch()


def extract_answer(pred_str, use_last_number=True):
    """从推理过程或模型输出中提取最后的答案。"""
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    else:
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]  # 提取最后的一个数字
            else:
                pred = ""
        else:
            pred = ""
    
    return pred.strip()

def parse_digits(num):
    """移除逗号，解析成浮点数"""
    num = re.sub(",", "", str(num))
    try:
        return float(num)
    except ValueError:
        if num.endswith("%"):
            num = num[:-1]
            try:
                return float(num) / 100
            except ValueError:
                pass
    return None

def numeric_equal(prediction: float, reference: float) -> bool:
    """比较两个浮点数是否在允许误差范围内相等"""
    return isclose(prediction, reference, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def math_equal(prediction: str, reference: str, include_percentage: bool=True) -> bool:
    """
    用于比较预测值和参考值是否相同，先尝试数值相等，再尝试符号相等。
    """
    # 1. 数值相等比较
    if evaluator.eq(prediction, reference):
        return True
    
    if parse_digits(prediction) is not None and parse_digits(reference) is not None:
        prediction_num = parse_digits(prediction)
        reference_num = parse_digits(reference)
        # 如果允许百分比，考虑转换百分比进行比较
        if include_percentage:
            gt_result = [reference_num / 100, reference_num, reference_num * 100]
        else:
            gt_result = [reference_num]
        
        for item in gt_result:
            if numeric_equal(prediction_num, item):
                return True

    # 2. 符号相等比较
    if symbolic_equal(prediction, reference):
        return True

    return False

def analyze_entropy_strategies(data: dict) -> Dict[str, float]:
    """
    分析给定目录下的JSON文件，使用不同的权重策略计算熵值
    
    策略组合：
    - 深度权重：递增(increasing)、不变(uniform)、递减(decreasing)
    - 路径选择：所有路径(all)、最长路径(longest)
    """
    
    def get_depth_weight(depth: int, strategy: str) -> float:
        """计算不同深度权重策略下的权重"""
        if strategy == "increasing":
            return depth + 1
        elif strategy == "square":
            return depth ** 2 + 1
        elif strategy == "uniform":
            return 1.0
        elif strategy == "decreasing":
            return 1.0 / (depth + 1)
        else:
            raise ValueError(f"Unknown weight strategy: {strategy}")
    
    def get_longest_paths(tree: Dict) -> List[List[Tuple[float, int]]]:
        """获取所有最长的路径，每个节点包含(entropy, depth)"""
        paths = []
        
        def dfs(node: Dict, current_path: List[Tuple[float, int]]):
            new_path = current_path + [(node.get('entropy', 0), node.get('depth', 0))]
            
            if not node.get('children'):
                paths.append(new_path)
            else:
                for child in node['children']:
                    dfs(child, new_path)
        
        dfs(tree, [])
        
        # 找到最长路径的长度
        max_length = max(len(path) for path in paths)
        # 只保留最长的路径
        longest_paths = [path for path in paths if len(path) == max_length]
        
        return longest_paths
    
    def get_final_paths(tree: Dict) -> List[List[Tuple[float, int]]]:
        """
        获取树中被保留的k条路径（按熵值排序）
        
        Args:
            tree: 字典形式的树结构
            
        Returns:
            List[List[Tuple[float, int]]]: 列表的列表，每个内部列表代表一条路径
            每个路径包含多个元组，元组中包含(熵值, 深度)
        """
        def _traverse(node: Dict, current_path: List[Tuple[float, int]], paths: List[List[Tuple[float, int]]]):
            # 添加当前节点到路径
            current_path = current_path + [(node.get("entropy", 0.0), node["depth"])]
            
            # 如果是叶子节点或没有子节点，保存路径
            if not node["children"] or node.get("is_answer", False):
                paths.append(current_path)
                return
                
            # 遍历所有子节点
            for child in node["children"]:
                _traverse(child, current_path[:], paths)
        
        paths = []
        _traverse(tree, [], paths)
        
        # 按最后一个非答案节点的熵值排序
        def get_sort_key(path):
            return path[-1][0] if path else float('-inf')
        
        sorted_paths = sorted(paths, key=get_sort_key, reverse=True)
        return sorted_paths

    def get_answer_paths(tree: Dict) -> List[List[Tuple[float, int]]]:
        """
        获取所有到达答案的路径
        
        Args:
            tree: 字典形式的树结构
            
        Returns:
            List[List[Tuple[float, int]]]: 列表的列表，每个内部列表代表一条到答案的路径
            每个路径包含多个元组，元组中包含(熵值, 深度)
        """
        def _traverse(node: Dict, current_path: List[Tuple[float, int]], paths: List[List[Tuple[float, int]]]):
            # 添加当前节点到路径
            current_path = current_path + [(node.get("entropy", 0.0), node["depth"])]
            
            # 如果是答案节点，保存路径
            if node.get("is_answer", False):
                paths.append(current_path)
                return
                
            # 如果不是答案节点且有子节点，继续遍历
            if node["children"]:
                for child in node["children"]:
                    _traverse(child, current_path[:], paths)
        
        answer_paths = []
        _traverse(tree, [], answer_paths)
        return answer_paths
    
    def get_all_paths(tree: Dict) -> List[List[Tuple[float, int]]]:
        """获取所有路径，包括被剪枝的路径"""
        paths = []
        
        def dfs(node: Dict, current_path: List[Tuple[float, int]]):
            new_path = current_path + [(node.get('entropy', 0), node.get('depth', 0))]
            
            if not node.get('children'):
                paths.append(new_path)
            else:
                for child in node['children']:
                    dfs(child, new_path)
        
        dfs(tree, [])
        return paths
    
    def calculate_weighted_entropy(paths: List[List[Tuple[float, int]]], weight_strategy: str) -> float:
        """计算给定路径和权重策略下的加权熵"""
        if not paths:
            return 0.0
            
        total_weighted_entropy = 0
        total_weight = 0
        
        for path in paths:
            path_entropy = 0
            path_weight = 0
            
            for entropy, depth in path:
                if entropy > 0:  # 只考虑非零熵
                    weight = get_depth_weight(depth, weight_strategy)
                    path_entropy += entropy * weight
                    path_weight += weight
            
            if path_weight > 0:
                total_weighted_entropy += path_entropy / path_weight
                total_weight += 1
        
        return total_weighted_entropy / max(1, total_weight)
    
    all_paths = get_all_paths(data)
    final_paths = get_final_paths(data)
    answer_paths = get_answer_paths(data)
    
    results = {}
    
    # 计算不同策略下的熵
    for weight_strategy in ["increasing", "uniform", "square"]:
        results[f"{weight_strategy}_all"] = calculate_weighted_entropy(all_paths, weight_strategy)
        results[f"{weight_strategy}_final"] = calculate_weighted_entropy(final_paths, weight_strategy)
        results[f"{weight_strategy}_answer"] = calculate_weighted_entropy(answer_paths, weight_strategy)
    
    return results

def get_clusters(candidates: List[str]) -> List[List[str]]:
    clusters = []
    
    for i in range(len(candidates)):
        # 查找匹配的簇
        matched_cluster = None
        for cluster in clusters:
            if math_equal(candidates[i], cluster[0]):
                matched_cluster = cluster
                break
        
        # 添加到已有簇或创建新簇
        if matched_cluster:
            matched_cluster.append(candidates[i])
        else:
            clusters.append([candidates[i]])
            
    return clusters

def calculate_entropy(clusters: List[List[str]]) -> float:
    cluster_sizes = [len(cluster) for cluster in clusters]
    total_size = sum(cluster_sizes)
    cluster_probs = [size / total_size for size in cluster_sizes]
    entropy = -sum(p * np.log(p + 1e-12) for p in cluster_probs)
    return entropy

def process_results(input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result = {}
                
                # 获取已有的生成答案
                generated_answers = [extract_answer(i) for i in data['direct']]
                
                # 计算熵值
                clusters = get_clusters(generated_answers)
                entropy = calculate_entropy(clusters)
                result['base_entropy'] = entropy
                
                # 计算正确率
                ground_truth = data["ground_truth"]
                correct_count = sum(1 for ans in generated_answers if math_equal(ans, ground_truth))
                accuracy = correct_count / len(generated_answers)
                result['accuracy'] = accuracy
                
                # 随机选择一个答案判断是否正确
                random_answer = random.choice(generated_answers)
                result['correct'] = math_equal(random_answer, ground_truth)
                
                result["re"] = analyze_entropy_strategies(data["tree"])
                
                print(result)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                    
                print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def main():
    input_folder =  "/home/test/test03/bohan/RE/base19"
    output_folder = "/home/test/test03/bohan/RE/base20"
    process_results(input_folder, output_folder)

if __name__ == "__main__":
    main()