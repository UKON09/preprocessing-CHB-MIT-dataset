"""
Copyright (c) 2025 UKON09. All rights reserved.
Project Name: preprocess_CHB-MIT_dataset
Author: UKON09
Project Version: v1.1.0
Python Version: 3.10
Created: 2025/4/7
GitHub: https: https://github.com/UKON09/preprocessing-CHB-MIT-dataset
"""

from matplotlib import rcParams
import os
import pandas as pd
from preprocess_data import preprocess_data


def main_preprocess_data(time_table_path, file_dir, prev_save_dir):
    # 存在非法电极(空电极)，这里可以不处理，因为pick channels会选择目标电极 drop 不需要的电极
    invalid_chb = ['chb14', 'chb16', 'chb17', 'chb20', 'chb21', 'chb22']
    empty_channels = ['--0', '--1', '--2', '--3', '--4']

    # 读取发病时间表，结构如下：
    # file（index）     num       start       end       duration
    # chb06_01          1         1724       1738        14
    annotations_df = pd.read_csv(time_table_path, index_col="file")

    if not os.path.exists(prev_save_dir):
        os.makedirs(prev_save_dir)  # 如果目录不存在，创建目录
        print("保存目录不存在，现已建立")

    dirs = os.listdir(file_dir)
    # 遍历目录下的所有文件和子目录
    for dir_name in dirs:
        subdir_path = os.path.join(file_dir, dir_name)
        preprocess_data(subdir_path, prev_save_dir, annotations_df)

def main():
    rcParams['font.family'] = 'SimHei'  # 黑体
    rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    # 调用函数时传入路径参数，如果没有按照我给的格式，可替换为自己的路径
    time_table_path = "../seizures_info.csv"  # 患者发病发病起止时间表，可替换路径
    file_dir = "../CHB-MIT_dataset"  # 数据文件所在目录
    prev_save_dir = "../CHB-MIT_pre_dataset/preictal"  # 保存的目录

    main_preprocess_data(time_table_path, file_dir, prev_save_dir)


if __name__ == '__main__':
    main()