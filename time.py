from datetime import datetime, timedelta
import csv

# 定义起始日期
start_date = datetime(2022, 1, 9)

# 定义周期长度（以天为单位）
cycle_length_days = 12

# 定义每个周期内的图片数量
images_per_cycle = 729

# 计算周期的数量
end_date = datetime(2022, 4, 3)  # 假设的结束日期
current_date = start_date
num_cycles = 0

while current_date <= end_date:
    num_cycles += 1
    current_date += timedelta(days=cycle_length_days)

# 生成时间戳
timestamps = []
folder_paths = {}
for cycle_index in range(num_cycles):
    # 计算每个周期的起始日期
    cycle_start_date = start_date + timedelta(days=cycle_index * cycle_length_days)

    for image_index in range(images_per_cycle):
        # 为每个周期内的每张图片生成时间戳
        timestamp = (cycle_start_date + timedelta(hours=image_index // 729 * 24)).strftime("%Y-%m-%d %H:%M")
        timestamps.append(timestamp)
        # 定义文件夹位置
        folder_name = cycle_start_date.strftime("%Y%m%d") + "_" + (cycle_start_date + timedelta(days=11)).strftime(
            "%Y%m%d")
        folder_paths[timestamp] = f"C:\\Users\\ZYX\\Desktop\\CDDFuse-main\\code\\data\\train\\opt\\{folder_name}\\"


file_path = r"C:\Users\ZYX\Desktop\CDDFuse-main\code\data\timestamps.csv"
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for timestamp in timestamps:
        writer.writerow([timestamp, folder_paths[timestamp]])
    print("Successfully saved to", file_path)
