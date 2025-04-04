import os
from PIL import Image

def read_images_in_batches(opt_folder_path, sar_folder_path, lbl_folder_path):
    opt_images = []
    sar_images = []
    lbl_images = []

    # 获取所有文件夹并按名称排序
    opt_subfolders = sorted([f for f in os.listdir(opt_folder_path) if os.path.isdir(os.path.join(opt_folder_path, f))], key=int)
    sar_subfolders = sorted([f for f in os.listdir(sar_folder_path) if os.path.isdir(os.path.join(sar_folder_path, f))], key=int)
    lbl_subfolders = sorted([f for f in os.listdir(lbl_folder_path) if os.path.isdir(os.path.join(lbl_folder_path, f))], key=int)

    if not opt_subfolders or not sar_subfolders or not lbl_subfolders:
        print("Error: No subfolders found in opt, sar, or lbl directories.")
        return opt_images, sar_images, lbl_images

    # 遍历文件夹顺序
    for opt_subfolder, sar_subfolder, lbl_subfolder in zip(opt_subfolders, sar_subfolders, lbl_subfolders):
        # 获取当前文件夹内的所有图片，按数字顺序排序
        opt_files = sorted([f for f in os.listdir(os.path.join(opt_folder_path, opt_subfolder)) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
        sar_files = sorted([f for f in os.listdir(os.path.join(sar_folder_path, sar_subfolder)) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
        lbl_files = sorted([f for f in os.listdir(os.path.join(lbl_folder_path, lbl_subfolder)) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

        # 确保每个文件夹内图片数量一致
        if len(opt_files) != len(sar_files) or len(sar_files) != len(lbl_files):
            print(f"Error: Mismatch in image count between folders in subfolder {opt_subfolder}.")
            continue

        # 遍历当前文件夹内的图片
        for opt_file, sar_file, lbl_file in zip(opt_files, sar_files, lbl_files):
            opt_file_path = os.path.join(opt_folder_path, opt_subfolder, opt_file)
            sar_file_path = os.path.join(sar_folder_path, sar_subfolder, sar_file)
            lbl_file_path = os.path.join(lbl_folder_path, lbl_subfolder, lbl_file)

            try:
                # 打开图片并确认读取成功
                with Image.open(opt_file_path) as opt_img, Image.open(sar_file_path) as sar_img, Image.open(lbl_file_path) as lbl_img:
                    opt_images.append(opt_file_path)
                    sar_images.append(sar_file_path)
                    lbl_images.append(lbl_file_path)
                    print(f"Read opt image: {opt_file_path}")
                    print(f"Read sar image: {sar_file_path}")
                    print(f"Read lbl image: {lbl_file_path}")
            except Exception as e:
                print(f"Error reading image {opt_file_path} or {sar_file_path}: {e}")

    return opt_images, sar_images, lbl_images


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