import json

# 定义文件路径
file1_path = '/home/new/桌面/intern/InternVL/vqa-en/intern_train_hor.jsonl'
file2_path = '/home/new/桌面/intern/InternVL/vqa-en/ocrvqa.jsonl'
merged_output_path = '/home/new/桌面/intern/InternVL/vqa-en/intern_train_hor2.jsonl'

# 打开第一个文件并读取数据
data = []
with open(file1_path, 'r', encoding='utf-8') as file1:
    for line in file1:
        data.append(json.loads(line.strip()))

# 打开第二个文件并读取数据
with open(file2_path, 'r', encoding='utf-8') as file2:
    for line in file2:
        data.append(json.loads(line.strip()))

# 将数据写入到新的 JSONL 文件中
with open(merged_output_path, 'w', encoding='utf-8') as outfile:
    for entry in data:
        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("两个 JSONL 文件已成功合并并保存到 merged_output.jsonl 文件中。")
