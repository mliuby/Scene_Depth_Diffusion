import os
import csv

# 文件夹 a 和 b 的路径
folder_a_path = "NYU_rawDepth/nyu_images"
folder_b_path = "NYU_rawDepth/nyu_rawDepths"

# 获取文件夹 a 和 b 中的文件名称列表
folder_a_files = os.listdir(folder_a_path)
folder_b_files = os.listdir(folder_b_path)

# 将文件名字写入 CSV 文件
output_file_path = "nyu2_train.csv"

with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Folder A', 'Folder B'])
    for i in range(max(len(folder_a_files), len(folder_b_files))):
        if i < len(folder_a_files):
            folder_a_file ="data/nyu_images/"+ folder_a_files[i]
        else:
            folder_a_file = ""
        if i < len(folder_b_files):
            folder_b_file = "data/nyu_rawDepths/"+folder_b_files[i]
        else:
            folder_b_file = ""
        writer.writerow([folder_a_file, folder_b_file])