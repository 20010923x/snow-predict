import os
import numpy as np
from PIL import Image
import torch


def read_images_in_batches(opt_folder_path, sar_folder_path, lbl_folder_path, num_regions=4, num_images_per_region=30):
    opt_images = []
    sar_images = []
    lbl_images = []

    # 获取所有文件夹并按名称排序
    opt_subfolders = sorted([f for f in os.listdir(opt_folder_path) if os.path.isdir(os.path.join(opt_folder_path, f))],
                            key=int)
    sar_subfolders = sorted([f for f in os.listdir(sar_folder_path) if os.path.isdir(os.path.join(sar_folder_path, f))],
                            key=int)
    lbl_subfolders = sorted([f for f in os.listdir(lbl_folder_path) if os.path.isdir(os.path.join(lbl_folder_path, f))],
                            key=int)

    # 获取实际的区域数（可以是 train 或 val 数据集中的区域数）
    actual_num_regions = min(len(opt_subfolders), len(sar_subfolders), len(lbl_subfolders), num_regions)
    print(f"Using {actual_num_regions} regions based on the available data.")

    if actual_num_regions == 0:
        print("Error: No regions found in any of the folders.")
        return opt_images, sar_images, lbl_images

    # 遍历每个区域文件夹
    for region_id in range(actual_num_regions):
        opt_subfolder = opt_subfolders[region_id]
        sar_subfolder = sar_subfolders[region_id]
        lbl_subfolder = lbl_subfolders[region_id]

        # 获取当前区域的所有图片，按文件名排序
        opt_files = sorted([f for f in os.listdir(os.path.join(opt_folder_path, opt_subfolder)) if f.endswith('.png')],
                           key=lambda x: int(x.split('.')[0]))
        sar_files = sorted([f for f in os.listdir(os.path.join(sar_folder_path, sar_subfolder)) if f.endswith('.png')],
                           key=lambda x: int(x.split('.')[0]))
        lbl_files = sorted([f for f in os.listdir(os.path.join(lbl_folder_path, lbl_subfolder)) if f.endswith('.png')],
                           key=lambda x: int(x.split('.')[0]))

        # 确保每个文件夹内图片数量一致且与 num_images_per_region 相等
        if len(opt_files) != num_images_per_region or len(sar_files) != num_images_per_region or len(lbl_files) != num_images_per_region:
            print(f"Error: Mismatch in image count for region {region_id + 1}. Expected {num_images_per_region} images.")
            continue

        # 存储该区域的 opt, sar, lbl 图像路径
        region_opt_images = []
        region_sar_images = []
        region_lbl_images = []

        for opt_file, sar_file, lbl_file in zip(opt_files, sar_files, lbl_files):
            opt_file_path = os.path.join(opt_folder_path, opt_subfolder, opt_file)
            sar_file_path = os.path.join(sar_folder_path, sar_subfolder, sar_file)
            lbl_file_path = os.path.join(lbl_folder_path, lbl_subfolder, lbl_file)

            try:
                # 读取图像并确认读取成功
                with Image.open(opt_file_path) as opt_img, Image.open(sar_file_path) as sar_img, Image.open(lbl_file_path) as lbl_img:
                    # 将PIL图像转换为NumPy数组并标准化
                    opt_img_np = np.array(opt_img, dtype=np.float32) / 255.0  # 归一化到[0,1]
                    sar_img_np = np.array(sar_img, dtype=np.float32) / 255.0
                    lbl_img_np = np.array(lbl_img, dtype=np.float32) / 255.0

                    # 将 NumPy 数组转换为 PyTorch 张量，并增加一个维度 (C, H, W)
                    region_opt_images.append(torch.tensor(opt_img_np).unsqueeze(0))  # C, H, W
                    region_sar_images.append(torch.tensor(sar_img_np).unsqueeze(0))  # C, H, W
                    region_lbl_images.append(torch.tensor(lbl_img_np).unsqueeze(0))  # C, H, W

                    print(f"Read opt image: {opt_file_path}")
                    print(f"Read sar image: {sar_file_path}")
                    print(f"Read lbl image: {lbl_file_path}")
            except Exception as e:
                print(f"Error reading image {opt_file_path} or {sar_file_path}: {e}")

        # 将该区域的图像拼接为 (30, C, H, W) 张量
        region_opt_tensor = torch.cat(region_opt_images, dim=0)  # 形状为 (30, C, H, W)
        region_sar_tensor = torch.cat(region_sar_images, dim=0)  # 形状为 (30, C, H, W)
        region_lbl_tensor = torch.cat(region_lbl_images, dim=0)  # 形状为 (30, C, H, W)

        # 将该区域的数据保存到总的图像列表
        opt_images.append(region_opt_tensor)
        sar_images.append(region_sar_tensor)
        lbl_images.append(region_lbl_tensor)

    # 将区域的数据堆叠成一个批次
    opt_batch = torch.stack(opt_images, dim=0)  # 形状为 (actual_num_regions, 30, C, H, W)
    sar_batch = torch.stack(sar_images, dim=0)  # 形状为 (actual_num_regions, 30, C, H, W)
    lbl_batch = torch.stack(lbl_images, dim=0)  # 形状为 (actual_num_regions, 30, C, H, W)

    return opt_batch, sar_batch, lbl_batch



# 指定文件夹路径
opt_folder_path = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\opt'
sar_folder_path = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\sar'
lbl_folder_path = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\lbl'

# 调用函数
opt_images, sar_images, lbl_images = read_images_in_batches(opt_folder_path, sar_folder_path, lbl_folder_path)

# 打印读取的图片数量
print(f"Total OPT images read: {len(opt_images)}")
print(f"Total SAR images read: {len(sar_images)}")
print(f"Total Label images read: {len(lbl_images)}")
