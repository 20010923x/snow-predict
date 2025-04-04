import os
import shutil
import subprocess


def split_data_sequentially(folder_paths, train_ratio, val_ratio, test_ratio):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    total_folders = len(folder_paths)
    train_end = int(total_folders * train_ratio)
    val_end = train_end + int(total_folders * val_ratio)

    train_folders = folder_paths[:train_end]
    val_folders = folder_paths[train_end:val_end]
    test_folders = folder_paths[val_end:]

    return train_folders, val_folders, test_folders


def create_symlinks(src_folders, dest_folder):
    for folder in src_folders:
        src_path = folder  # 源文件夹路径
        dest_path = os.path.join(dest_folder, os.path.basename(folder))  # 目标符号链接路径
        if not os.path.exists(dest_path):
            # 使用mklink命令创建符号链接
            command = f'mklink /D "{dest_path}" "{src_path}"'
            subprocess.run(command, shell=True)
        else:
            print(f"Destination path already exists: {dest_path}")

def write_paths_to_file(folder_paths, file_path):
    with open(file_path, 'w') as file:
        for folder in folder_paths:
            file.write(folder + '\n')
    print(f"Paths written to {file_path}")


def move_folders(folder_list, destination):
    for folder in folder_list:
        src_path = folder  # 源文件夹路径
        dest_path = os.path.join(destination, os.path.basename(folder))  # 目标文件夹路径
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)  # 移动文件夹
        else:
            print(f"Source folder does not exist: {src_path}")


# 示例使用
folder_paths = [r'E:\opt\20220109', r'E:\opt\20220121', r'E:\opt\20220202', r"E:\opt\20220214", r"E:\opt\20220226", r"E:\opt\20220310",
                r"E:\opt\20220322", r"E:\opt\20220403", r"E:\opt\20220415", r"E:\opt\20220427", r"E:\opt\20220509", r"E:\opt\20220521",
                r"E:\opt\20220602", r"E:\opt\20220614", r"E:\opt\20220626", r"E:\opt\20220708", r"E:\opt\20220720", r"E:\opt\20220801",
                r"E:\opt\20220825", r"E:\opt\20220906", r"E:\opt\20220918", r"E:\opt\20220930", r"E:\opt\20221012", r"E:\opt\20221024",
                r"E:\opt\20221105", r"E:\opt\20221117", r"E:\opt\20221129", r"E:\opt\20220214", r"E:\opt\20221211", r"E:\opt\20221223"]   # 替换为你的文件夹路径
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_folders, val_folders, test_folders = split_data_sequentially(folder_paths, train_ratio, val_ratio, test_ratio)

# 创建符号链接或复制文件夹
train_dest = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\opt'
val_dest = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\val\opt'
test_dest = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\test\opt'

create_symlinks(train_folders, train_dest)
create_symlinks(val_folders, val_dest)
create_symlinks(test_folders, test_dest)

move_folders(train_folders, train_dest)
move_folders(val_folders, val_dest)
move_folders(test_folders, test_dest)


# 将路径写入文件
write_paths_to_file(train_folders, r'E:\data\train_folders.txt')
write_paths_to_file(val_folders, r'E:\data\val_folders.txt')
write_paths_to_file(test_folders, r'E:\data\test_folders.txt')

print(f"Training set folders: {train_folders}")
print(f"Validation set folders: {val_folders}")
print(f"Test set folders: {test_folders}")