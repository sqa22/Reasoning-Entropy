from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import json


def analyze_entropy_with_decision_tree(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 存储每种指标的值
    base_entropy = []
    increasing_all = []
    correct_labels = []

    # 遍历文件夹，读取 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取 base_entropy 和 increasing_all 指标
                base_entropy.append(data.get("base_entropy"))
                re_metrics = data.get("re", {})
                increasing_all.append(re_metrics.get("increasing_all", None))

                # 提取 correct (布尔值)
                correct_labels.append(-data.get("correct"))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # 清理数据，去除 None 值
    clean_data = [
        (be, ia, correct) for be, ia, correct in zip(base_entropy, increasing_all, correct_labels)
        if be is not None and ia is not None
    ]
    if not clean_data:
        print("No valid data found.")
        return

    base_entropy, increasing_all, correct_labels = zip(*clean_data)

    # 构造特征矩阵和标签
    X = list(zip(base_entropy, increasing_all))  # 特征矩阵
    y = correct_labels  # 标签

    # 训练决策树分类器
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # 限制深度为3，防止过拟合
    clf.fit(X, y)

    # 计算 AUROC
    predictions = clf.predict_proba(X)[:, 1]  # 获取正类的概率
    try:
        decision_tree_auroc = roc_auc_score(y, predictions)
        print(f"AUROC for decision tree classifier: {decision_tree_auroc:.4f}")
    except ValueError as e:
        print(f"Not enough data to compute AUROC for decision tree: {e}")
        return

    # 可视化决策树
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=["Base Entropy", "Increasing All"], class_names=["False", "True"], filled=True)
    plt.title(f"Decision Tree (AUROC: {decision_tree_auroc:.4f})")
    
    # 保存决策树图表
    decision_tree_path = os.path.join(output_folder, "decision_tree_structure.png")
    plt.savefig(decision_tree_path, format='png', dpi=300)
    plt.close()
    print(f"Saved decision tree structure to {decision_tree_path}")


def main():
    input_folder = "/home/test/test03/bohan/RE/base20"  # 输入文件夹路径
    output_folder = "/home/test/test03/bohan/RE/plots"  # 输出图表存储路径
    analyze_entropy_with_decision_tree(input_folder, output_folder)


if __name__ == "__main__":
    main()