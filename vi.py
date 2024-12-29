import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def analyze_entropy_auroc(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 存储每种指标的值
    entropy_metrics = {
        "base_entropy": [],
        "increasing_all": [],
        "increasing_final": [],
        "increasing_answer": [],
        "uniform_all": [],
        "uniform_final": [],
        "uniform_answer": [],
        "square_all": [],
        "square_final": [],
        "square_answer": [],
    }
    correct_labels = []

    # 遍历文件夹，读取 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取 base_entropy 和 re 指标
                entropy_metrics["base_entropy"].append(data.get("base_entropy"))
                re_metrics = data.get("re", {})

                for metric in entropy_metrics.keys():
                    if metric != "base_entropy":
                        entropy_metrics[metric].append(re_metrics.get(metric, None))

                # 提取 correct (布尔值)
                correct_labels.append(-data.get("correct"))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # 检查是否存在有效数据
    if not correct_labels or not any(entropy_metrics.values()):
        print("No valid data found.")
        return

    # 遍历每个 entropy 指标，计算 AUROC 并保存图表
    for metric, values in entropy_metrics.items():
        # 清理数据，去除 None 值
        clean_data = [(v, correct) for v, correct in zip(values, correct_labels) if v is not None]
        metric_values, valid_labels = zip(*clean_data)

        # 计算 AUROC
        try:
            auroc = roc_auc_score(valid_labels, metric_values)
            print(f"AUROC for {metric}: {auroc:.4f}")
        except ValueError as e:
            print(f"Not enough data to compute AUROC for {metric}: {e}")
            continue

        # 可视化并保存 ROC 曲线
        plt.figure(figsize=(10, 6))
        plt.hist(
            [metric_values[i] for i in range(len(valid_labels)) if valid_labels[i]],
            bins=30,
            alpha=0.6,
            color="green",
            label="Correct=True"
        )
        plt.hist(
            [metric_values[i] for i in range(len(valid_labels)) if not valid_labels[i]],
            bins=30,
            alpha=0.6,
            color="red",
            label="Correct=False"
        )
        plt.title(f"Distribution of {metric} (AUROC: {auroc:.4f})")
        plt.xlabel(f"{metric}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()

        # 保存图表到文件
        output_path = os.path.join(output_folder, f"{metric}_distribution.png")
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()  # 关闭当前图表，释放内存
        print(f"Saved distribution plot for {metric} to {output_path}")


def main():
    input_folder = "/home/test/test03/bohan/RE/base20"  # 输入文件夹路径
    output_folder = "/home/test/test03/bohan/RE/plots"  # 输出图表存储路径
    analyze_entropy_auroc(input_folder, output_folder)


if __name__ == "__main__":
    main()