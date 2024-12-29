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

def init_model(gpu_id: int) -> LLM:
    """在指定GPU上初始化模型"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return LLM(
        model="/home/test/test03/models/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )

def process_questions(gpu_id: int, questions: List[dict], result_queue: mp.Queue):
    """单个进程处理问题的函数"""
    llm = init_model(gpu_id)
    
    sampling_params = SamplingParams(
        n=12,
        temperature=0.8,
        top_p=0.95,
        max_tokens = 4096,
    )
    
    max_attempts = 50
    results = []
    
    for question_data in tqdm(questions):
        question = "Answer the following question in latex and wrap the final answer in \\boxed{}. You should think step by step and separate different step with \\n\\n.\n" + question_data["question"]
        print(question)
        ground_truth = question_data["answer"]
        
        all_generated_solutions = []  # 存储所有生成的答案
        found_correct_answer = False
        
        for attempt in range(max_attempts):
            outputs = llm.generate([question], sampling_params)
            generated_solutions = [generated.text.strip() for generated in outputs[0].outputs]
            
            # 保存生成的答案
            all_generated_solutions.extend(generated_solutions)
            
            # 检查是否有一个答案是正确的
            for solution in generated_solutions:
                extracted_solution = extract_answer(solution)
                if math_equal(extracted_solution, ground_truth):
                    found_correct_answer = True
                    break
            
            if found_correct_answer:
                break  # 如果找到正确答案，停止生成
        
        print("Correct and Stop")
        
        # 将所有生成的答案保存到结果中
        result = {
            "question": question,
            "solutions": all_generated_solutions,
            "found_correct_answer": found_correct_answer
        }
        results.append(result)
    
    result_queue.put((gpu_id, results))

def main():
    data = []
    with open("Qwen2.5-Math/evaluation/data/gsm8k/test.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    data = [i for i in data if i["evel"] == "Level 5"]
    
    all_questions = data
    random.shuffle(all_questions)
    
    num_gpus = 8
    questions_per_gpu = len(all_questions) // num_gpus
    
    result_queue = mp.Queue()
    processes = []
    
    # 启动进程
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * questions_per_gpu
        end_idx = start_idx + questions_per_gpu if gpu_id < num_gpus - 1 else len(all_questions)
        
        gpu_questions = all_questions[start_idx:end_idx]
        
        p = mp.Process(
            target=process_questions,
            args=(gpu_id, gpu_questions, result_queue)
        )
        processes.append(p)
        p.start()
    
    # 收集所有结果
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 保存结果到JSON文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generation_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # 打印一些统计信息
    print(f"\nGeneration Summary:")
    print(f"Total questions processed: {len(all_results)}")
    print(f"Solutions per question: {len(all_results[0]['solutions'])}")
    
    # 示例输出第一个问题的部分结果
    print(f"\nSample output for first question:")
    print(f"Question: {all_results[0]['question']}")
    print(f"First 3 solutions:")
    for i, solution in enumerate(all_results[0]['solutions'][:3]):
        print(f"{i+1}. {solution}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()