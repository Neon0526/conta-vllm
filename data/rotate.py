from PIL import Image
import os

# 指定根文件夹路径
root_folder = '/home/new/桌面/bunny/Bunny/vqa_data_aug/'

# 遍历根文件夹下的所有文件夹
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    # 确保是文件夹
    if not os.path.isdir(folder_path):
        continue
    
    print(f'Processing images in folder: {folder_name}')
    
    # 构造要处理的图片名称
    image_name = f'{folder_name}.jpg'
    image_path = os.path.join(folder_path, image_name)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f'Image {image_name} not found in folder {folder_name}')
        continue
    
    # 打开图像
    try:
        image = Image.open(image_path)
    except OSError:
        print(f"Skipping file {image_name} due to unsupported mode")
        continue
    
    # 确保图像是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 定义要执行的旋转角度
    angles = [30, 60, 120, 150]
    
    # 旋转并保存图像
    for angle in angles:
        rotated_image = image.rotate(angle)
        rotated_image.save(os.path.join(folder_path, f'{folder_name}_rot{angle}.jpg'))

print("Image processing completed.")
