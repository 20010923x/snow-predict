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
                image_files.append(os.path.join(subdir, file))
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # 根据提取的数字对图片文件进行排序
    image_files.sort()

    # 重命名图片
    for index, file_path in enumerate(image_files, start=1):
        # 构建新的文件名，格式为 1_1, 1_2, 1_3, ..., 1_100
        new_file_name = f"10_{index}{os.path.splitext(file_path)[1]}"  # 使用固定前缀1和递增的编号
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

        try:
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"Renamed {file_path} to {new_file_path}")
        except Exception as e:
            print(f"Error renaming {file_path} to {new_file_path}: {e}")


# 指定文件夹路径
folder_path = r'C:\Users\ZYX\Desktop\未处理数据集\lbl\20220509'
# 调用函数
rename_images_in_folder(folder_path)