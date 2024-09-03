import numpy as np
import torch
import math
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm  # 引入tqdm库

# 原始代码中的一些参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def process_conversations(data, model, tokenizer, generation_config, output_json_file):
    results = []
    
    # 使用 tqdm 包裹数据列表以显示进度条
    for entry in tqdm(data, desc="Processing entries"):
        image_path = entry['image']
        conversation = entry['conversations']

        # 加载图像并准备输入
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

        for i in range(0, len(conversation), 2):
            question = conversation[i]['value']
            standard_answer = conversation[i+1]['text']

            # 模型生成响应
            response = model.chat(tokenizer, pixel_values, question, generation_config)

            # 保存结果
            results.append({
                "question": question,
                "standard_answer": standard_answer,
                "response": response
            })

    # 将结果保存到 JSON 文件中
    with open(output_json_file, 'w', encoding='utf-8') as output_json:
        json.dump(results, output_json, ensure_ascii=False, indent=4)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 加载模型和Tokenizer
path = "/home/new/桌面/intern/InternVL/internvl_chat/work_dirs/internvl_normal/merge_model_bgr_5"
device_map = 'auto'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 设置生成配置
generation_config = dict(max_new_tokens=24, do_sample=False)

# JSONL 文件路径和输出文件路径
jsonl_file = "/home/new/桌面/intern/InternVL/vqa-en/intern_test_new.jsonl"  # 替换为你的 JSONL 文件路径
output_json_file = "/home/new/桌面/intern/InternVL/internvl_chat/infer/epoch5/normal/bgr2_output_new.json"  # 替换为你的输出 JSON 文件路径

# 处理对话并保存结果
data = load_jsonl(jsonl_file)
process_conversations(data, model, tokenizer, generation_config, output_json_file)
