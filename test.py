import os
from PIL import Image
import numpy as np
import torch

def read_label_image(image_path, black_threshold, white_threshold):
    # 读取图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image_array = np.array(image)

    # 打印图像的灰度值范围和直方图
    print(f"Image min grayscale value: {image_array.min()}")
    print(f"Image max grayscale value: {image_array.max()}")
    print(f"Image mean grayscale value: {image_array.mean()}")

    # 将图像转换为二进制标签张量
    label_tensor = np.zeros_like(image_array, dtype=np.int32)
    label_tensor[image_array < black_threshold] = 0
    label_tensor[image_array > white_threshold] = 1

    # 打印一些标签值的统计信息
    print(f"Label tensor unique values: {np.unique(label_tensor)}")

    # 将numpy数组转换为torch张量
    label_tensor = torch.from_numpy(label_tensor).unsqueeze(0)  # 增加一个批次维度
    return label_tensor

# 调用函数
image_path = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\lbl\20220109\2.png'  # 替换为你的图像文件路径
black_threshold =0
white_threshold = 150
label_tensor = read_label_image(image_path, black_threshold, white_threshold)
print(f'Label tensor:\n{label_tensor}')

import matplotlib.pyplot as plt
import cv2

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.hist(image.ravel(), 256, [0, 256])
plt.show()