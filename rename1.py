
import os
import re

import os
import re

def rename_images_in_folder(folder_path):
    # 用于存储找到的图片文件及其对应的数字
    image_files = []

    # 遍历文件夹中的所有子文件夹
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否是图片（这里假设图片文件有常见的图片扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 提取文件名中的数字
                match = re.match(r'(\d+)_(\d+)', file)
                if match:
                    # 将文件路径和提取的数字存储在列表中
                    image_files.append((int(match.group(1)), int(match.group(2)), os.path.join(subdir, file)))

    # 根据提取的数字对图片文件进行排序
    image_files.sort()

    # 重命名图片
    for index, (group_num, sub_num, file_path) in enumerate(image_files, start=1):
        # 构建新的文件名
        new_file_name = f"{index}{os.path.splitext(file_path)[1]}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

        try:
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"Renamed {file_path} to {new_file_path}")
        except Exception as e:
            print(f"Error renaming {file_path} to {new_file_path}: {e}")

# 自动识别指定路径下的所有文件夹并处理每个文件夹
base_folder_path = r'C:\Users\ZYX\Desktop\CDDFuse-main原\code\data\train1\sar'

# 获取 base_folder_path 下的所有子文件夹
for folder_name in os.listdir(base_folder_path):
    folder_path = os.path.join(base_folder_path, folder_name)

    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        # 调用函数处理文件夹中的图片
        rename_images_in_folder(folder_path)
