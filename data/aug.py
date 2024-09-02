from PIL import Image, ImageOps
import os

# 指定根文件夹路径
root_folder = '/home/new/桌面/bunny/Bunny/vqa_data_aug/'

# 遍历根文件夹下的所有文件夹（假设这些文件夹包含图片）
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    # 确保是文件夹
    if not os.path.isdir(folder_path):
        continue
    
    print(f'Processing images in folder: {folder_name}')
    
    # 循环处理每张图片
    for image_name in os.listdir(folder_path):
        if not image_name.endswith('_mir.jpg'):
            continue
        
        # 打开原始图像
        image_path = os.path.join(folder_path, image_name)
        try:
            image = Image.open(image_path)
        except OSError:
            print(f"Skipping file {image_name} due to unsupported mode")
            continue
        
        # 转换图像模式为RGB或灰度模式
        if image.mode != 'RGB' and image.mode != 'L':
            image = image.convert('RGB')  # 或者 'L'，根据需要选择合适的模式
        
        # # 1. 水平翻转
        # flipped_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
        # flipped_horizontal.save(os.path.join(folder_path, f'{image_name[:-4]}_hor.jpg'))
        
        # # 2. 垂直翻转
        # flipped_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
        # flipped_vertical.save(os.path.join(folder_path, f'{image_name[:-8]}_mir.jpg'))
        
        # # 3. 镜像翻转（水平+垂直）
        # mirrored_image = ImageOps.mirror(image)
        # mirrored_image.save(os.path.join(folder_path, f'{image_name[:-4]}_mir.jpg'))
        
        # 4. 旋转（假设旋转90度）
        rotated_image = image.rotate(30)
        rotated_image.save(os.path.join(folder_path, f'{image_name[:-4]}_rot30.jpg'))
        
        # # 5. RGB通道改变（假设改为BGR顺序）
        # bgr_image = Image.merge('RGB', (image.split()[2], image.split()[1], image.split()[0]))
        # bgr_image.save(os.path.join(folder_path, f'{image_name[:-4]}_bgr.jpg'))

print("Image processing completed.")
