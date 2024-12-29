import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def find_optimal_threshold(fpr, tpr, thresholds):
    """找到最优阈值（约登指数最大点）"""
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def evaluate_with_threshold(values, labels, threshold):
    """使用给定阈值评估性能"""
    predictions = (values >= threshold).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

def check_class_balance(labels):
    """检查数据集中的类别分布"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    return len(unique_labels), dict(zip(unique_labels, counts))

def analyze_entropy_auroc(input_folder, output_folder, test_size=0.2, random_state=42):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

    # 读取数据
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                entropy_metrics["base_entropy"].append(data.get("base_entropy"))
                re_metrics = data.get("re", {})

                for metric in entropy_metrics.keys():
                    if metric != "base_entropy":
                        entropy_metrics[metric].append(re_metrics.get(metric, None))

                label = 0 if data.get("correct") == -1 else 1
                correct_labels.append(label)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not correct_labels or not any(entropy_metrics.values()):
        print("No valid data found.")
        return

    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    results = {}

    for metric, values in entropy_metrics.items():
        print(f"\nProcessing metric: {metric}")
        
        # 清理数据
        valid_data = [(v, l) for v, l in zip(values, correct_labels) if v is not None]
        if not valid_data:
            print(f"Skipping {metric} - no valid data")
            continue
        
        metric_values, valid_labels = zip(*valid_data)
        metric_values = np.array(metric_values)
        valid_labels = np.array(valid_labels)

        # 检查类别分布
        n_classes, class_dist = check_class_balance(valid_labels)
        print(f"Class distribution for {metric}:", class_dist)
        
        if n_classes < 2:
            print(f"Skipping {metric} - only one class present in the data")
            results[metric] = {
                'error': 'Only one class present in the data',
                'class_distribution': class_dist
            }
            continue

        try:
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                metric_values, valid_labels, 
                test_size=test_size, 
                random_state=random_state,
                stratify=valid_labels
            )

            # 检查训练集和测试集的类别分布
            _, train_dist = check_class_balance(y_train)
            _, test_dist = check_class_balance(y_test)
            
            if len(train_dist) < 2 or len(test_dist) < 2:
                print(f"Skipping {metric} - insufficient class distribution after split")
                results[metric] = {
                    'error': 'Insufficient class distribution after split',
                    'train_distribution': train_dist,
                    'test_distribution': test_dist
                }
                continue

            # 计算ROC曲线和性能指标
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                
                fpr_train, tpr_train, thresholds_train = roc_curve(y_train, X_train)
                optimal_threshold = find_optimal_threshold(fpr_train, tpr_train, thresholds_train)
                train_auroc = roc_auc_score(y_train, X_train)
                test_auroc = roc_auc_score(y_test, X_test)

            train_metrics = evaluate_with_threshold(X_train, y_train, optimal_threshold)
            test_metrics = evaluate_with_threshold(X_test, y_test, optimal_threshold)

            results[metric] = {
                'optimal_threshold': float(optimal_threshold),
                'train_auroc': float(train_auroc),
                'test_auroc': float(test_auroc),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_distribution': train_dist,
                'test_distribution': test_dist
            }

            print(f"Optimal threshold (from training): {optimal_threshold:.4f}")
            print(f"Train AUROC: {train_auroc:.4f}")
            print(f"Test AUROC: {test_auroc:.4f}")
            print("Training metrics:", train_metrics)
            print("Test metrics:", test_metrics)

            # 绘制训练集分布图
            plt.figure(figsize=(10, 6))
            plt.hist(
                X_train[y_train == 1], bins=30, alpha=0.6,
                color="green", label=f"Correct=True (n={sum(y_train == 1)})"
            )
            plt.hist(
                X_train[y_train == 0], bins=30, alpha=0.6,
                color="red", label=f"Correct=False (n={sum(y_train == 0)})"
            )
            plt.axvline(x=optimal_threshold, color='black', linestyle='--',
                       label=f'Threshold: {optimal_threshold:.4f}')
            plt.title(f"Training Distribution of {metric}\n(AUROC: {train_auroc:.4f})")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(train_folder, f"{metric}_distribution.png"))
            plt.close()

            # 绘制测试集分布图
            plt.figure(figsize=(10, 6))
            plt.hist(
                X_test[y_test == 1], bins=30, alpha=0.6,
                color="green", label=f"Correct=True (n={sum(y_test == 1)})"
            )
            plt.hist(
                X_test[y_test == 0], bins=30, alpha=0.6,
                color="red", label=f"Correct=False (n={sum(y_test == 0)})"
            )
            plt.axvline(x=optimal_threshold, color='black', linestyle='--',
                       label=f'Threshold: {optimal_threshold:.4f}')
            plt.title(f"Test Distribution of {metric}\n(AUROC: {test_auroc:.4f})")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(test_folder, f"{metric}_distribution.png"))
            plt.close()

        except Exception as e:
            print(f"Error processing {metric}: {e}")
            results[metric] = {'error': str(e)}
            continue

    # 保存结果
    try:
        with open(os.path.join(output_folder, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    input_folder = "/home/test/test03/bohan/RE/base20"
    output_folder = "/home/test/test03/bohan/RE/plots"
    analyze_entropy_auroc(input_folder, output_folder)

if __name__ == "__main__":
    main()