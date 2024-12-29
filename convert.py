import json

# 文件路径
input_path = "/home/test/test03/bohan/RE/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl"
output_path = "/home/test/test03/bohan/RE/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl"

# 读取数据
data = []
with open(input_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

# 修改数据
for d in data:
    d["answer"] = d["answer"].split("####")[-1].strip()

# 写入数据
with open(output_path, "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")