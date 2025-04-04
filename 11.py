import os
import numpy as np
from PIL import Image
import cv2



folder_path = r"C:\Users\ZYX\Desktop\CDDFuse-main原\code\data\train1\opt\118"

save_folder_path = r"C:\Users\ZYX\Desktop\CDDFuse-main原\code\data\train1\118"

# 获取文件夹中所有图像文件的路径
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 定义黑色和白色像素的阈值
black_threshold = 160
white_threshold = 250
# 定义要标记的颜色
black_mark_color = (0, 0, 0)
white_mark_color = (255, 255, 255)

for image_file in image_files:
    # 加载原图和预测图
    original_image = Image.open(image_file).convert('RGBA')

    file_name = os.path.basename(image_file)
    prediction_image_file = os.path.join(save_folder_path, os.path.splitext(file_name)[0] + '.png')

    # 假设预测图的文件名与原图相同，但扩展名为'.png'
    #prediction_image_file = image_file.replace('.png', '.png')
    prediction_image = Image.open(prediction_image_file).convert('RGB')

    # 确保原图和预测图的尺寸为256x256
    assert original_image.size == (256, 256)
    assert prediction_image.size == (256, 256)

    # 将图像转换为numpy数组
    original_array = np.array(original_image)
    prediction_array = np.array(prediction_image)

    # 将原图转换为灰度图
    original_array_gray = cv2.cvtColor(original_array[:, :, :3], cv2.COLOR_RGB2GRAY)


# 检测原图中的黑色和白色像素，并在预测图上标记
    for i in range(256):
        for j in range(256):
            pixel_value = original_array_gray[i, j]
            if pixel_value < black_threshold:
                prediction_array[i, j] = black_mark_color
            elif pixel_value > white_threshold:
                prediction_array[i, j] = white_mark_color

    # 将修改后的预测图转换回图像
    marked_prediction_image = Image.fromarray(prediction_array)

# 保存并显示新图像
    marked_prediction_image_path = prediction_image_file
    marked_prediction_image.save(marked_prediction_image_path)
