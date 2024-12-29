import torch
from vllm import LLM, SamplingParams
import multiprocessing as mp
from typing import List
import os
os.environ["OUTLINES_CACHE_DIR"] = f"/tmp/outlines_cache_{os.getpid()}"

import json
from datetime import datetime
import re
from math import isclose
import sympy as sp
from tqdm import tqdm
import random
import hashlib

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy import latex2sympy

from transformers import AutoTokenizer

from symeval import EvaluatorMathBatch

evaluator = EvaluatorMathBatch()

# 定义特殊字符（如中文）的正则模式
special_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

MODEL_ID = "/home/test/test03/models/Meta-Llama-3.1-8B-Instruct"

demo = r"""
Question: In the diagram below, $\\overline{AB}\\parallel \\overline{CD}$ and $\\angle AXE$ is $108^\\circ$ less than 3 times $\\angle CYX$.  Find $\\angle BXY$.\n\n[asy]\n\nunitsize(1inch);\n\npair A,B,C,D,X,Y,EE,F;\n\nA = (0,0);\n\nB=(1,0);\n\nC = (0,0.8);\n\nD=(1,0.8);\n\nEE = (0.35,-0.3);\n\nF = (0.8,1.1);\n\ndraw(EE--F);\n\ndraw(A--B);\n\ndraw(C--D);\n\ndot(A);\n\ndot(B);\n\ndot(C);\n\ndot(D);\n\ndot(EE);\n\ndot(F);\n\nlabel(\"$E$\",EE,S);\n\nlabel(\"$F$\",F,N);\n\nX = intersectionpoint(A--B,EE--F);\n\nY = intersectionpoint(C--D,EE--F);\n\nlabel(\"$X$\",X,NNW);\n\nlabel(\"$Y$\",Y,NNW);\n\nlabel(\"$A$\",A,W);\n\nlabel(\"$B$\",B,E);\n\nlabel(\"$C$\",C,W);\n\nlabel(\"$D$\",D,E);\n\ndot(X);\n\ndot(Y);\n\n[/asy]
Answer:
## Step1:
When two parallel lines are cut by a transversal:
   * $\\overline{AB}\\parallel\\overline{CD}$ implies $\\angle AXE = \\angle CYX$

## Step2:
Let $x = \\angle AXE$
   * Given: $x$ is $108°$ less than $3x$
   * Equation: $x = 3x - 108°$
   * $x - 3x = -108°$
   * $-2x = -108°$
   * $x = 54°$

## Step3:
$\\angle BXY = \\angle AXE$ (vertically opposite angles)
   * Therefore, $\\angle BXY = $\\boxed{54^\\circ}$, the answer is $\\boxed{54^\\circ}$.


Question: On the island of Mumble, the Mumblian alphabet has only $5$ letters, and every word in the Mumblian language has no more than $3$ letters in it. How many words are possible? (A word can use a letter more than once, but $0$ letters does not count as a word.)
Answer: 
## Step1:
Let's define what we need to count:
   * Words can be 1, 2, or 3 letters long
   * Each position can use any of the 5 letters
   * Letters can be repeated

## Step2:
For 1-letter words:
   * Each position has 5 choices
   * Total: 5 words

## Step3:
For 2-letter words:
   * First position: 5 choices
   * Second position: 5 choices
   * By multiplication principle: $5 \times 5 = 25$ words

## Step4:
For 3-letter words:
   * First position: 5 choices
   * Second position: 5 choices
   * Third position: 5 choices
   * By multiplication principle: $5 \times 5 \times 5 = 125$ words

## Step5:
Total number of possible words:
   * Sum of words of all lengths
   * $5 + 25 + 125 = 155$
Thus, the answer is $\\boxed{155}$.


Question: Every June 1, an ecologist takes a census of the number of wrens in a state park. She noticed that the number is decreasing by $40\\%$ each year. If this trend continues, in what year will the census show that the number of wrens is less than $10\\%$ of what it was on June 1, 2004?
Answer: 
## Step1:
Let's say the initial population in 2004 is P
* After 1 year: 0.6P (60% of P)
* After 2 years: 0.6 \\times 0.6P = 0.36P
* After 3 years: 0.6 \\times 0.6 \\times 0.6P = 0.216P
* The pattern is: After n years: (0.6)ⁿP

## Step2:
We want to find when the population becomes less than 10% of P
* (0.6)ⁿP < 0.1P
* (0.6)ⁿ < 0.1

## Step3:
Using logarithms:
* ln(0.6)ⁿ < ln(0.1)
* n \\times ln(0.6) < ln(0.1)
* n > ln(0.1)/ln(0.6)
* n > -2.303/-0.511
* n > 4.51

## Step4:
Since n must be a whole number of years and needs to be the first year where population is less than 10%:
* n = 5
* Starting from 2004, this means 2009

Therefore, in 2009, the census will first show that the wren population is less than 10% of what it was in 2004.

The answer is $\\boxed{2009}$.


Question: Four concentric circles are drawn with radii of 1, 3, 5 and 7. The inner circle is painted black, the ring around it is white, the next ring is black and the outer ring is white. What is the ratio of the black area to the white area? Express your answer as a common fraction.
Answer: 
## Step1:
Let's identify the areas:
* Inner black circle: radius $= 1$
* First white ring: between radius $1$ and $3$
* Second black ring: between radius $3$ and $5$
* Outer white ring: between radius $5$ and $7$

## Step2:
Calculate the area of each ring using $\pi r^2$:
* Inner black circle: $\pi(1)^2 = \pi$
* First white ring: $\pi(3)^2 - \pi(1)^2 = 9\pi - \pi = 8\pi$
* Second black ring: $\pi(5)^2 - \pi(3)^2 = 25\pi - 9\pi = 16\pi$
* Outer white ring: $\pi(7)^2 - \pi(5)^2 = 49\pi - 25\pi = 24\pi$

## Step3:
Total black area:
* Black $= \text{Inner circle} + \text{Second black ring}$
* Black $= \pi + 16\pi = 17\pi$

## Step4:
Total white area:
* White $= \text{First white ring} + \text{Outer white ring}$
* White $= 8\pi + 24\pi = 32\pi$

## Step5:
Ratio of black to white:
* Black:White $= 17\pi:32\pi$
* Simplify by dividing both by $\pi$
* $= 17:32$

Therefore, the ratio of black area to white area is $17:32$, which as a fraction is $\\frac{17}{32}$.

The answer is $\\boxed{\\frac{17}{32}}$.


Question: Compute the product of $0.\\overline{123}$ and $9$, and write your result as a fraction in simplified form.
Answer:
## Step1:
First, let's write $0.\overline{123}$ as a fraction
* Let $x = 0.\overline{123}$
* Then $1000x = 123.\overline{123}$
* $1000x = 123 + x$
* $999x = 123$
* $x = \\frac{123}{999} = \\frac{41}{333}$

## Step2:
Now multiply by 9:
* $\\frac{41}{333} \\times 9$
* $= \\frac{41 \times 9}{333}$
* $= \\frac{369}{333}$
* $= \\frac{123}{111}$

## Step3:
Since 123 and 111 are both divisible by 3:
* $= \\frac{41}{37}$

The answer is $\\boxed{\\frac{41}{37}}$.
"""

def init_model(gpu_id: int) -> LLM:
    """在指定GPU上初始化模型"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return LLM(
        model = MODEL_ID
    )

def generate_tree(llm, sampling_params, prefix, depth, max_depth=20):
    """生成推理树，限制最大深度为 max_depth，并在出现特殊字符时剪枝。"""
    if depth >= max_depth:
        return {"prefix": prefix, "child": [], "leaf": 0, "depth": depth + 1}

    node = {"prefix": prefix, "child": [], "leaf": 0, "depth": depth + 1}
    
    print(node)
    
    # 检查是否有特殊字符（如中文），如果有则剪枝
    if special_char_pattern.search(prefix):
        print(f"剪枝 - 发现特殊字符: {prefix}")
        return node  # 剪枝
    
    outputs = llm.generate([prefix], sampling_params)
    solutions = [generated.text.strip() for generated in outputs[0].outputs]
    
    for sol in solutions:
        if "\\boxed" in sol or "\boxed" in sol or "the answer is" in sol:
            node["child"].append({"prefix": prefix + sol, "child": [], "leaf": 1, "depth": depth + 1})
        else:
            node["child"].append(generate_tree(llm, sampling_params, prefix + sol + "Step", depth + 1, max_depth))
    
    return node

def process_questions(gpu_id: int, questions: List[dict], output_dir: str, lock: mp.Lock):
    """单个进程处理问题的函数，每道题保存到一个新的文件中"""
    llm = init_model(gpu_id)
    
    sampling_params = SamplingParams(
        n=8,
        temperature=0.8,
        top_p=0.95,
        max_tokens=4096,
        stop = ["Step", "<|eot_id|>"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    for question_data in tqdm(questions, desc=f"GPU {gpu_id}"):
        ground_truth = question_data["answer"]
        
        messages = [
            {"role": "system", "content": "Answer the following problem in LaTeX and put your final answer in \\boxed{}."},
            {"role": "user", "content": question_data["problem"] + f"Examples:\n\n{demo}\n\nNow begin!\n\nQuestion: {question_data['problem']}\n\nAnswer: "}
        ]
        
        question = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        tree = generate_tree(llm, sampling_params, question, 0, max_depth=20)
        
        # 创建每道题的单独文件
        question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        output_file = os.path.join(output_dir, f"{question_hash}.json")
        
        # 使用锁来确保文件写入的原子性
        result = {
            "gpu_id": gpu_id,
            "question": question,
            "tree": tree,
            "ground_truth": ground_truth
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    data = []
    with open("Qwen2.5-Math/evaluation/data/math/test.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    all_questions = data
    random.shuffle(all_questions)
    
    num_gpus = 8
    questions_per_gpu = len(all_questions) // num_gpus
    
    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"generation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
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
            args=(gpu_id, gpu_questions, output_dir, file_lock)
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print(f"Results saved to directory: {output_dir}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()