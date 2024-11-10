import os
import shutil

# 定义根目录
root_path = r'H:\DataSet\infrared_small_target\NUDT-MIRSDT'
new_extension = ['.jpg','.png','.png','.mat']
# 获取所有文件夹名称
dir_names = os.listdir(root_path)
for dir_name in dir_names:
    file_path = os.path.join(root_path, dir_name)
    dir_names_ins = os.listdir(file_path)
    i = 0
    for dir_name_in in dir_names_ins:
        i += 1
        source_path = os.path.join(file_path, dir_name_in)
        target_path = os.path.join(root_path,dir_name_in,dir_name)
        # 创建目标文件夹
        os.makedirs(target_path, exist_ok=True)
        # 将原始文件夹中的文件复制到目标文件夹并修改文件后缀
        for file_name in os.listdir(source_path):
            source_file = os.path.join(source_path, file_name)
            target_file = os.path.join(target_path, os.path.splitext(file_name)[0] + new_extension[i-1])
            shutil.move(source_file, target_file)