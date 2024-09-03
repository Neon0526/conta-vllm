import json
import pandas as pd
import os
from PIL import Image

file_path = '/home/new/桌面/intern/InternVL/vqa-en/output.json'
file_path2 = '/home/new/桌面/intern/InternVL/vqa-en/output.jsonl'
output_file_path = '/home/new/桌面/intern/InternVL/vqa-en/intern_data.jsonl'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

output_data = []
for idx, entry in enumerate(data):
    image_path = entry['image']
    conversations = entry['conversations']
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        width, height = 0,0
    new_conversations = []
    for i,conv in enumerate(conversations):
        if conv['from'] == 'human' and i == 0:
            new_conversations.append({"from":"human",
                                      "value":f"<image>\n{conv['value']}"})
        else:
            new_conversations.append({'from':conv["from"],
                                      "value":conv["value"]})
    transformed_entry = {
        "id": idx,
        "image": image_path,
        "width": width,
        "height": height,
        "conversations": new_conversations
    }

    output_data.append(transformed_entry)

with open(file_path2, 'r',encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line.strip())
        output_data.append(entry)
# print(output_data)

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in output_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')