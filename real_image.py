import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_labels_for_feature_prediction(base_dir, num_folders, num_images_per_folder, target_size=(64, 64)):

    # 获取文件夹路径并按文件夹名称排序
    folder_paths = [os.path.join(base_dir, folder) for folder in sorted(os.listdir(base_dir), key=int) if
                    os.path.isdir(os.path.join(base_dir, folder))]

    all_labels = []

    # 确保文件夹数量与区域数相符
    if len(folder_paths) != num_folders:
        raise ValueError(f"The number of folders ({len(folder_paths)}) must match the num_folders ({num_folders})")

    # 遍历每个区域文件夹
    for region_idx in range(num_folders):  # 遍历每个区域
        label_images = []

        # 从每个文件夹读取图片（每个文件夹对应一个区域）
        folder_path = folder_paths[region_idx]
        for img_idx in range(num_images_per_folder):  # 遍历每个图片编号
            label_image_path = os.path.join(folder_path, f"{img_idx + 1}.png")

            if os.path.exists(label_image_path):
                label_img = Image.open(label_image_path).convert('L')  # 标签是单通道灰度图
                label_img = label_img.resize(target_size, Image.Resampling.LANCZOS)
                label_img = np.array(label_img) / 255.0  # 归一化操作
                label_images.append(np.expand_dims(label_img, axis=0))  # [1, height, width]
            else:
                raise FileNotFoundError(f"Label image not found at path: {label_image_path}")

        # 将区域内的图像堆叠
        all_labels.append(np.stack(label_images, axis=0))  # [num_images_per_folder, 1, height, width]

    # 将所有区域的标签堆叠到一个数组中
    all_labels = np.stack(all_labels, axis=0)  # [num_folders, num_images_per_folder, 1, height, width]

    print(f"Labels shape: {all_labels.shape}")
    #print(all_labels)
    return all_labels



def visualize_labels(real_snow_images, num_images_to_show=5):
    """
    可视化读取到的标签图像。

    :param real_snow_images: 标签数组，形状为 [num_folders, num_images_per_folder, 1, height, width]
    :param num_images_to_show: 要展示的标签图像的数量
    """
    num_folders, num_images_per_folder, _, height, width = real_snow_images.shape

    for i in range(min(num_images_to_show, num_folders)):
        # 遍历展示前 num_images_to_show 个区域的图像
        plt.figure(figsize=(15, 5))

        for j in range(num_images_per_folder):
            plt.subplot(1, num_images_per_folder, j + 1)
            label_img = real_snow_images[i, j, 0, :, :]  # 获取区域 i 对应的第 j 张标签图像
            plt.imshow(label_img, cmap='gray')
            plt.title(f"Region {i + 1} - Image {j + 1}")
            plt.axis('off')

        plt.show()


# 假设你有正确的标签路径和区域数量
base_dir = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\lbl'
num_folders = 1  # 区域数量（文件夹数）
num_images_per_folder = 390  # 每个文件夹中的图片数量

# 读取标签数据
real_snow_images = load_labels_for_feature_prediction(base_dir, num_folders, num_images_per_folder)

# 假设 real_snow_images 已经被正确读取
visualize_labels(real_snow_images, num_images_to_show=1)


