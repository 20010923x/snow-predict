import os
import cv2
import torch
def process_image_to_tensor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

    # 二值化
    _, binary_img = cv2.threshold(img, 50, 250, cv2.THRESH_BINARY)

    # 将二值化后的图像转换为张量
    binary_img_tensor = torch.tensor(binary_img, dtype=torch.float32).unsqueeze(0) / 255.0

    # 将白色部分（积雪）转换为 1，黑色部分（陆地）转换为 0
    labels_tensor = (binary_img_tensor > 0.5).int()

    return  labels_tensor

def get_tensors_from_images(root_dir):
    label_tensors = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件扩展名
                image_path = os.path.join(subdir, file)
                tensor = process_image_to_tensor(image_path)
                label_tensors.append(tensor)  # 将张量添加到列表

    return label_tensors

