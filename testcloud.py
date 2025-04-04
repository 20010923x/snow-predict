import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt


# 加载云掩码元数据
metadata = pd.read_csv(r'C:\Users\ZYX\Desktop\greenearthnet-main\cloudmask\models-master\cloudSEN12\data\cloudsen12_metadata.csv')

# 筛选出标记为 high（有云）的数据
high_cloud_data = metadata[metadata ['label_type'] == 'high']
print(high_cloud_data)
# 加载光学和 SAR 图像
opt_image_path = r'F:\1snow\tu\opt\20220521.tif'
sar_image_path = r'F:\1snow\tu\sar旧\20220521.tif'

# 使用 Rasterio 打开图像
opt_image = rasterio.open(opt_image_path).read(1)  # 读取光学图像第一个波段
sar_image = rasterio.open(sar_image_path).read(1)  # 读取 SAR 图像第一个波段
# 假设 cloud_mask 是根据 label_type == 'high' 提取的掩码

cloud_mask = np.loadtxt(r'C:\Users\ZYX\Desktop\greenearthnet-main\cloudmask\models-master\cloudSEN12\data\cloudsen12_metadata.csv', delimiter=',')  # 示例云掩码文件
opt_image_masked = np.where(cloud_mask == 1, np.nan, opt_image)

# 可视化原始图像和屏蔽后的光学图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Optical Image")
plt.imshow(opt_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Masked Optical Image")
plt.imshow(opt_image_masked, cmap='gray')
plt.show()