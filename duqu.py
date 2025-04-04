import os
from PIL import Image


def read_images_in_batches(folder_path):
    read_images = []
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # 确保有24个子文件夹
    if len(subfolders) != 24:
        print(f"Error: There should be exactly 24 subfolders, but found {len(subfolders)}.")
        return

    # 读取图片的编号，假设编号从1开始，每个文件夹内有729张图片
    for i in range(1, 730):  # 从1到729
        file_name = f"{i}.png"
        all_images_found = True

        for subfolder in subfolders:
            file_path = os.path.join(folder_path, subfolder, file_name)

            if os.path.exists(file_path):
                try:
                    # 打开图片并确认读取成功
                    with Image.open(file_path) as img:
                        read_images.append(file_path)
                        print(f"Read image: {file_path}")
                except Exception as e:
                    print(f"Error reading image {file_path}: {e}")
            else:
                all_images_found = False
                break  # 如果当前编号的图片在某个子文件夹中不存在，跳出内层循环

        if not all_images_found:
            print(f"Not all subfolders contain image {file_name}.")
            break  # 如果某个子文件夹没有当前编号的图片，提前结束

    return read_images


# 指定文件夹路径
folder_path = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\opt'
# 调用函数
all_read_images = read_images_in_batches(folder_path)

# 打印读取的图片数量
print(f"Total images read: {len(all_read_images)}")