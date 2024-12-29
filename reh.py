import json
import os
import random
import hashlib
from datetime import datetime
import multiprocessing as mp
from typing import List, Dict
import numpy as np
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer
from collections import defaultdict

print("!!!")

demo = """
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

# 定义语义聚类逻辑
def get_containment(expr1, expr2, llm):
    if expr1 == expr2:
        return True
    
    """
    判断两个表达式的语义关系：等价或蕴涵。

    Args:
        expr1 (str): 第一个表达式。
        expr2 (str): 第二个表达式。
        llm (LLM): 模型对象。
        sampling_params (SamplingParams): 采样参数。

    Returns:
        bool: True 如果 expr1 包含 expr2 或等价，False 否则。
    """
    prompt = f"""You are a precise mathematical reasoning assistant. Given two mathematical statements, determine if one statement implies or is equivalent to the other. 

Output format: Return only "contains" or "not_contains"
- Return "contains" if:
  * The statements are equivalent
  * One statement logically implies or includes the other
- Return "not_contains" if:
  * The statements are different or unrelated
  * Neither statement implies the other

Examples:
Statement 1: "For all real numbers x, x² ≥ 0"
Statement 2: "For all real numbers x, x² + 1 > 0"
Output: contains

Statement 1: "The triangle is isosceles and right-angled"
Statement 2: "The triangle is isosceles"
Output: contains

Statement 1: "The function is continuous"
Statement 2: "The sequence converges"
Output: not_contains

Now analyze the following pair of statements:
Statement 1: {expr1}
Statement 2: {expr2}
Output: """
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=10,
    )
    result = ""
    while result not in ["contains", "not_contains"]:
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        result = outputs[0].outputs[0].text.strip().split()[0].strip()
        return result == "contains"

def semantic_clustering(candidates, llm):
    clusters = []
    
    for i in range(len(candidates)):
        # 查找匹配的簇
        matched_cluster = None
        for cluster in clusters:
            if get_containment(candidates[i], cluster[0], llm):
                matched_cluster = cluster
                break
        
        # 添加到已有簇或创建新簇
        if matched_cluster:
            matched_cluster.append(candidates[i])
        else:
            clusters.append([candidates[i]])
            
    return clusters

def calculate_semantic_entropy_with_clustering(candidates, clusters):
    cluster_sizes = [len(cluster) for cluster in clusters]
    print(cluster_sizes)
    total_size = sum(cluster_sizes)
    cluster_probs = [size / total_size for size in cluster_sizes]
    entropy = -sum(p * np.log(p + 1e-12) for p in cluster_probs)
    return entropy

class Node:
    def __init__(self, content, depth, prefix, parent=None):
        self.content = content
        self.depth = depth
        self.parent = parent
        self.prefix = prefix
        self.children = []
        self.entropy = 0.0
        self.is_answer = False
        self.clusters = []  # Add clusters field
        self.next_step_contents = []  # Add next step contents field

def tree_to_dict(node):
    return {
        "prefix": node.prefix,
        "content": node.content,
        "depth": node.depth,
        "entropy": node.entropy,
        "is_answer": node.is_answer,
        "clusters": [list(cluster) for cluster in node.clusters],  # Save clusters
        "next_step_contents": node.next_step_contents,  # Save next step contents
        "children": [tree_to_dict(child) for child in node.children]
    }

def generate_and_prune_tree(llm, sampling_params, question, max_depth=10, k=5):
    root = Node("", 0, question)
    current_level = [root]
    
    for depth in range(max_depth):
        next_level = []
        
        print(f"depth: {depth}")
        
        for parent_node in current_level:
            if parent_node.is_answer:
                continue

            print(f"Prefix{parent_node.prefix[-200:]}")
            outputs = llm.generate([parent_node.prefix + f"Step{depth + 1}"], sampling_params, use_tqdm=False)
            child_contents = [output.text.strip() for output in outputs[0].outputs]
            print("child_contents is", child_contents) 
             
            child_nodes = [Node(content, depth + 1, parent_node.prefix + content + f"Step{depth + 2}", parent_node) for content in child_contents]
            parent_node.children = child_nodes
            next_level.extend(child_nodes)
        
        if not next_level:
            break
        
        for node in next_level:
            if "\\boxed" in node.content or "the answer is" in node.content:
                node.is_answer = True
                continue
                
            outputs = llm.generate([node.prefix], sampling_params, use_tqdm=False)
            next_step_contents = [output.text.strip() for output in outputs[0].outputs]
            node.next_step_contents = next_step_contents  # Save next step contents
            
            # Save clusters before calculating entropy
            clusters = semantic_clustering(next_step_contents, llm)
            node.clusters = clusters  # Save clusters
            node.entropy = calculate_semantic_entropy_with_clustering(next_step_contents, clusters)
        
        next_level.sort(key=lambda x: x.entropy)
        pruned_level = next_level[-k:]
        
        current_level = pruned_level
    
    return root

def process_questions(gpu_id, questions, output_dir, file_lock, k):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", use_tqdm=False)
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    sampling_params = SamplingParams(n=16, temperature=0.7, max_tokens=2048, stop = ["Step", "<|eot_id|>"])

    for question_data in questions:
        ground_truth = question_data.get("answer", None)
        
        messages = [
            {"role": "system", "content": "Answer the following problem in LaTeX and put your final answer in \\boxed{}. You should think by step."},
            {"role": "user", "content": question_data["problem"] + f"Examples:\n\n{demo}\n\nNow begin!\n\nQuestion: {question_data['problem']}\n\nAnswer: "}
        ]
        
        question = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成推理树
        tree = generate_and_prune_tree(llm, sampling_params, question, max_depth=10, k=k)
        
        # 转换为字典格式并保存
        tree_dict = tree_to_dict(tree)
        
        question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        output_file = os.path.join(output_dir, f"{question_hash}.json")

        with file_lock:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "question": question,
                    "tree": tree_dict,
                    "ground_truth": ground_truth
                }, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run semantic entropy-based reasoning")
    parser.add_argument("--k", type=int, default=5, help="Number of paths to keep per level during pruning.")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use.")
    args = parser.parse_args()

    # 加载问题数据
    data = []
    with open("Qwen2.5-Math/evaluation/data/math/test.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    data = [i for i in data if i["level"] == "Level 5"]
    all_questions = data
    random.shuffle(all_questions)
    
    print(len(all_questions))

    questions_per_gpu = len(all_questions) // args.num_gpus

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"generation_results_{timestamp}_k_{args.k}"
    os.makedirs(output_dir, exist_ok=True)

    file_lock = mp.Lock()
    processes = []

    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * questions_per_gpu
        end_idx = start_idx + questions_per_gpu if gpu_id < args.num_gpus - 1 else len(all_questions)
        gpu_questions = all_questions[start_idx:end_idx]

        p = mp.Process(
            target=process_questions,
            args=(gpu_id + 0 , gpu_questions, output_dir, file_lock, args.k)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Results saved to directory: {output_dir}")

if __name__ == "__main__":
    main()