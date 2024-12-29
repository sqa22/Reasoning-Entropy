import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import re
from math import isclose
import sympy as sp
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy import latex2sympy

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

def count_attempts_until_correct(solutions, ground_truth):
    """统计需要多少次尝试才能得到正确答案"""
    for i, solution in enumerate(solutions, 1):
        extracted = extract_answer(solution)
        if math_equal(extracted, ground_truth):
            return i
    return None

def analyze_results(results_file):
    # 读取结果文件
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 统计每个问题需要的尝试次数
    attempts_counts = []
    hard_questions = []  # 存储需要10次以上尝试的问题
    
    for result in tqdm(results):
        question_data = result['question'].split('\n')[1]  # 获取实际问题
        # 从原始数据文件中查找答案
        with open("Qwen2.5-Math/evaluation/data/gsm8k/test.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                if data['question'] == question_data:
                    ground_truth = data['answer']
                    break
        
        count = count_attempts_until_correct(result['solutions'], ground_truth)
        if count is not None:
            attempts_counts.append(count)
            if count > 10:
                hard_questions.append({
                    "question": question_data,
                    "attempts_needed": count,
                    "ground_truth": ground_truth,
                    "all_solutions": result['solutions']
                })
    
    # 计算累积正确率
    max_attempts = max(attempts_counts)
    correct_rates = []
    attempts_range = range(1, max_attempts + 1)
    
    total_questions = len(attempts_counts)
    for i in attempts_range:
        correct_count = sum(1 for x in attempts_counts if x <= i)
        correct_rate = correct_count / total_questions
        correct_rates.append(correct_rate)
    
    # 绘制曲线（使用log10刻度）
    plt.figure(figsize=(10, 6))
    plt.semilogx(attempts_range, correct_rates, 'b-')
    plt.xlabel('Number of Attempts (log scale)')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Accuracy vs Number of Attempts')
    plt.grid(True)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'accuracy_curve_log_{timestamp}.png')
    plt.close()
    
    # 保存困难问题
    with open(f'hard_questions_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(hard_questions, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"Total questions analyzed: {total_questions}")
    print(f"Average attempts needed: {np.mean(attempts_counts):.2f}")
    print(f"Median attempts needed: {np.median(attempts_counts):.2f}")
    print(f"Maximum attempts needed: {max_attempts}")
    print(f"Number of hard questions (>10 attempts): {len(hard_questions)}")

# 使用示例
analyze_results("generation_results_20241112_192256.json")