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

def process_single_gpu(gpu_id: int, file_list: List[str], input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    sampling_params = SamplingParams(n=64, temperature=0.7, max_tokens=4096)
    
    for filename in file_list:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            outputs = llm.generate([data["question"]], sampling_params, use_tqdm=False)
            
            data['direct'] = [output.text.strip() for output in outputs[0].outputs]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            print(f"GPU {gpu_id} successfully processed {filename}")
            
        except Exception as e:
            print(f"GPU {gpu_id} error processing {filename}: {str(e)}")

def main():
    input_folder = "/home/test/test03/bohan/RE/generation_results_20241209_005956_k_5"
    output_folder = "/home/test/test03/bohan/RE/base19"
    
    # 获取所有json文件
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    # 将文件平均分配给7个GPU
    n_gpus = 7
    files_per_gpu = len(all_files) // n_gpus
    file_splits = []
    
    for i in range(n_gpus):
        start_idx = i * files_per_gpu
        end_idx = start_idx + files_per_gpu if i < n_gpus - 1 else len(all_files)
        file_splits.append(all_files[start_idx:end_idx])
    
    # 创建进程
    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=process_single_gpu,
            args=(gpu_id, file_splits[gpu_id], input_folder, output_folder)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()