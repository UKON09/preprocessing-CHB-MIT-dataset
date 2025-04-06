import mne
from mne.annotations import Annotations
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data

matplotlib.use('TkAgg')
plt.ion()
# 存在非法电极(空电极)，这里可以不处理，因为pick channels会选择目标电极 drop 不需要的电极
invalid_chb = ['chb14', 'chb16', 'chb17', 'chb20', 'chb21', 'chb22']
empty_channels = ['--0', '--1', '--2', '--3', '--4']

# 调用函数时传入路径参数
time_table_path = "seizures_info.csv"  # 患者发病发病起止时间表
file_dir = "./CHB-MIT_dataset"  # 数据文件所在目录
save_dir = r"E:\data_preprocessing\CHB-MIT\CHB-MIT_dataset_npy"  # 保存的目录

# 读取发病时间表，结构如下：
# file（index）     num       start       end       duration
# chb06_01          1         1724       1738        14
annotations_df = pd.read_csv(time_table_path, index_col="file")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 如果目录不存在，创建目录
    print("保存目录不存在，现已建立")


dirs = os.listdir(file_dir)
# 遍历目录下的所有文件和子目录
for dir_name in dirs:
    subdir_path = os.path.join(file_dir, dir_name)
    preprocess_data(subdir_path, save_dir, annotations_df)

