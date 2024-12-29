import torch
from vllm import LLM, SamplingParams
import multiprocessing as mp
from typing import List
import os
import json
from datetime import datetime
import re
from math import isclose
import sympy as sp
from tqdm import tqdm
import random

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy import latex2sympy

from transformers import AutoTokenizer

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
                return float(num) / 64
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
    try:
        if evaluator.eq(prediction, reference):
            return True
    except:
        pass
    
    if parse_digits(prediction) is not None and parse_digits(reference) is not None:
        prediction_num = parse_digits(prediction)
        reference_num = parse_digits(reference)
        # 如果允许百分比，考虑转换百分比进行比较
        if include_percentage:
            gt_result = [reference_num / 64, reference_num, reference_num * 64]
        else:
            gt_result = [reference_num]
        
        for item in gt_result:
            if numeric_equal(prediction_num, item):
                return True

    # 2. 符号相等比较
    if symbolic_equal(prediction, reference):
        return True

    return False

def init_model(gpu_id: int) -> LLM:
    """在指定GPU上初始化模型"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )

def process_questions(gpu_id: int, questions: List[dict], output_file: str, lock: mp.Lock):
    """单个进程处理问题的函数"""
    llm = init_model(gpu_id)
    
    sampling_params = SamplingParams(
        n=128,
        temperature=0.8,
        top_p=0.95,
        max_tokens = 4096,
    )
    
    max_attempts = 5000
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
    
    for question_data in tqdm(questions, desc=f"GPU {gpu_id}"):
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question_data["question"]}
        ]
        
        question = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"GPU {gpu_id}: Processing question:", question)
        ground_truth = question_data["answer"]
        
        correct_solutions = []
        found_correct_answers = 0
        total_attempts = 0
        
        while found_correct_answers < 64 and total_attempts < max_attempts:
            if total_attempts > 2000 and found_correct_answers < 1:
                break
            
            outputs = llm.generate([question], sampling_params)
            generated_solutions = [generated.text.strip() for generated in outputs[0].outputs]
            total_attempts += len(generated_solutions)
            
            # 检查答案是否正确并保存
            for solution in generated_solutions:
                extracted_solution = extract_answer(solution)
                if math_equal(extracted_solution, ground_truth):
                    correct_solutions.append(solution)
                    found_correct_answers += 1
            if found_correct_answers >= 64:
                break
        
        print(f"GPU {gpu_id}: Found {found_correct_answers} correct answers after {total_attempts} attempts")
        
        # 将结果保存到jsonl文件
        result = {
            "gpu_id": gpu_id,
            "question": question,
            "correct_solutions": correct_solutions,
            "num_correct": found_correct_answers,
            "total_attempts": total_attempts,
            "ground_truth": ground_truth
        }
        
        # 使用锁来确保文件写入的原子性
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    data = []
    with open("Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    # data = [i for i in data if i["level"] in ["Level 5"]]
    
    all_questions = data
    random.shuffle(all_questions)
    
    num_gpus = 7
    questions_per_gpu = len(all_questions) // num_gpus
    
    # 创建输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generation_results_{timestamp}.jsonl"
    
    # 创建锁对象
    file_lock = mp.Lock()
    
    processes = []
    
    # 启动进程
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * questions_per_gpu
        end_idx = start_idx + questions_per_gpu if gpu_id < num_gpus - 1 else len(all_questions)
        
        gpu_questions = all_questions[start_idx:end_idx]
        
        p = mp.Process(
            target=process_questions,
            args=(gpu_id + 1, gpu_questions, output_file, file_lock)
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 计算并打印总体统计信息
    total_questions = 0
    total_correct_10 = 0
    total_attempts = 0
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            total_questions += 1
            if result['num_correct'] >= 64:
                total_correct_10 += 1
            total_attempts += result['total_attempts']
    
    print(f"\nGeneration Summary:")
    print(f"Results saved to: {output_file}")
    print(f"Total questions processed: {total_questions}")
    print(f"Questions with 64 correct answers: {total_correct_10}")
    print(f"Average attempts per question: {total_attempts/total_questions:.2f}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()