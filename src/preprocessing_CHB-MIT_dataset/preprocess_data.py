"""
Copyright (c) 2025 UKON09. All rights reserved.
Project Name: preprocess_CHB-MIT_dataset
Author: UKON09
Project Version: v1.1.0
Python Version: 3.10
Created: 2025/4/7
GitHub: https: https://github.com/UKON09/preprocessing-CHB-MIT-dataset
"""

import mne
from mne.annotations import Annotations
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import time


def extract_file(file_dir, extension=None, need_extension=True):
    """
    获取目标路径下的文件名

    :param file_dir：目标文件所在的路径
    :param extension：目标文件后缀，例如名为'xx.edf'的'.edf' (默认为 None)
    :param need_extension: 是否需要保留名字中的后缀 (默认为 True)

    :return files(list): 包含文件名的列表
    """
    files = []
    if extension != None:
        if need_extension == True:
            files = [f for f in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, f)) and f.endswith(extension)]
        else:
            files = [os.path.splitext(f)[0] for f in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, f)) and f.endswith(extension)]
    else:
        if need_extension == True:
            files = [os.path.splitext(f)[0] for f in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, f))]
        else:
            raise Exception("Exepect a file extension.")
    return files


def seizure_annotations(raw, annotations_df, file_str):
    """
    读取 .edf 文件并添加癫痫发作的标注

    :param raw: .edf 数据文件
    :param annotations_df(DataFrame): 注释信息
    :param file_str: 无后缀患者文件名
    """

    # 筛选出索引为 file_str 的所有行
    filtered_df = annotations_df[annotations_df.index == file_str]
    # 获取行数。num = len(annotations_df.loc[file_str])如果只有一行，则会输出file_str有几列
    num = len(filtered_df)
    for i in range(1, num+1):
        # 解析标注信息
        onsets = filtered_df[(filtered_df['num'] == i)]['start'].values[0]
        durations = filtered_df[(filtered_df['num'] == i)]['duration'].values[0]
        descriptions = 1 #seizures

        # 创建 Annotations 对象
        annotations = Annotations(onsets, durations, descriptions)

        # 将标注添加到 Raw 对象
        raw.set_annotations(annotations)


def apply_filter(raw):
    """
    滤波器设计来自论文：
    Yikai Yang et al 2023 Neuromorph. Comput. Eng. 3 014010
    - 50 Hz噪声：滤除频率范围在 47–53 Hz 和 97–103 Hz 之间的信号。
    - 60 Hz噪声：滤除频率范围在 57–63 Hz 和 117–123 Hz 之间的信号。
    https://doi.org/10.1016/j.seizure.2019.08.006
    常用的滤波器有：巴斯特滤波器、小波变化、傅里叶变换
    https://doi.org/10.1016/j.artmed.2025.103095
    常见的滤波范围

    :param raw: MNE Raw 数据对象
    """
    # 高通滤波器
    raw.filter(l_freq=0.1, h_freq=None, method='iir', iir_params=dict(order=5, ftype='butter'))
    # 高通滤波器
    raw.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=dict(order=6, ftype='butter'))
    # 低通滤波器
    raw.filter(l_freq=None, h_freq=70, method='iir', iir_params=dict(order=5, ftype='butter'))
    # 60Hz工频干扰
    raw.filter(l_freq=61, h_freq=58, method='fir', fir_design='firwin', phase='zero-double', filter_length='auto',
               l_trans_bandwidth=1, h_trans_bandwidth=1)
    # 50Hz工频干扰
    # raw.filter(l_freq=51, h_freq=48, method='iir', iir_params=dict(order=1, ftype='butter'))


def cut_segments(raw, prev_file_path, start_time, duration):
    """
    根据注释的开始时间和持续时间来切割数据。
    若注释的开始时间 > 提取片段长度，则从 (start_time - duration, start_time) 进行切割。
    若注释的开始时间 <= 提取片段长度，则先将该片段与上一片段连接，再裁剪 (start_time - duration, start_time)。

    :param raw (mne.io.Raw): 原始数据对象
    :param prev_file_path: 前一个.edf文件地址

    :return segments (mne.io.Raw): 切割后的数据片段
    """

    # 获取注释信息

    # 用来存储切割的片段
    segment = None
    # 如果 start_time > duration，则直接切割当前片段
    if start_time > duration:
        segment_start = start_time - duration
        segment_end = start_time
        segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)

    else:
        # 如果 start_time <= duration，连接当前片段和前一个片段
        prev_raw = mne.io.read_raw_edf(prev_file_path, preload=True, verbose=False)
        prev_end_time = prev_raw.times[-1]# 上一个片段的结束时间
        if prev_end_time+start_time <= duration:
            prev_segment_start = 0
            prev_segment_end = prev_end_time
            prev_segment = prev_raw.copy().crop(tmin=prev_segment_start, tmax=prev_segment_end)
            segment_start = 0
            segment_end = start_time
            cur_segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
            segment = mne.concatenate_raws([prev_segment, cur_segment])
            warnings.warn(f'上一个片段时长不足，最后总片段长度为：{segment.times[-1]}')
            time.sleep(3)
        else:
            prev_segment_start = prev_end_time - duration + start_time
            prev_segment_end = prev_end_time
            prev_segment = prev_raw.copy().crop(tmin=prev_segment_start, tmax=prev_segment_end)
            segment_start = 0
            segment_end = start_time
            cur_segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
            segment = mne.concatenate_raws([prev_segment, cur_segment])

    return segment


def set_montage(raw, channels, positions):
    '''
    由于电极名称设置不规范，所以将大部分电极位映射到 1020标准电极位， T7-FT9 和 FT10-T8 单独计算，后 montage
    以下是映射信息，其中 001 为 T7、FT9 电极空间坐标的中值，002 为 FT10、T8 电极空间坐标的中值
    new_channels = ['AF7', 'FT7', 'TP7', 'PO7',
                    'AF3', 'FC3', 'CP3', 'PO3',
                    'AF4', 'FC4', 'CP4', 'PO4',
                    'AF8', 'FT8', 'TP8', 'PO8',
                    'FCz', 'CPz', '001', 'Nz', '002']

    参数：
    :param raw: MNE Raw 数据对象
    :param channels: 双极导联电极名称表
    '''

    new_positions = [[num / 2000 for num in sub_list] for sub_list in positions]
    channels_positions = dict(zip(channels, new_positions))
    montage = mne.channels.make_dig_montage(ch_pos=channels_positions, coord_frame='head')
    raw.set_montage(montage)


def remove_artefacts(raw, n_components=16, threshold=0.9):
    """
    使用 ICA 和 ICLabel 自动去除伪迹成分（如眼动、心电图伪迹）。

    :param raw: 原始EEG数据（Raw对象）
    :param n_components: ICA分解的成分数（默认为20）
    :param threshold: 用于判定伪迹成分的概率阈值（默认为0.5）

    :return raw_clean: 去除伪迹后的EEG数据（Raw对象）
    """
    # 1. 运行 ICA 分解EEG数据
    raw_copied = raw.copy()
    raw_copied = raw_copied.set_eeg_reference("average")
    ica = ICA(
        n_components=n_components,
        max_iter="auto",
        method="infomax",
        random_state=90,
        fit_params=dict(extended=True),
    )
    ica.fit(raw_copied)
    print('拟合完成，自动剔除高概率组分')

    # 使用label_components函数标记ICA成分
    data = label_components(raw_copied, ica, method='iclabel')
    labels = data['labels']
    probs = data['y_pred_proba']
    # label_to_prob = dict(zip(data['labels'], data['y_pred_proba']))
    # 定义要清除成分的标签列表和阈值
    artifact_labels = ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']
    # 自动排查
    exclude_idx = []
    for i in range(len(labels)):
        label = labels[i]
        prob = probs[i]
        print(f'{i}的成分是 {label} 的概率为 {prob}')
        if label in artifact_labels:
            if prob >= threshold:
                exclude_idx.append(i)
                print(f'{i}已排除')
            else:
                print(f'{i}不排除')
    length = len(exclude_idx)
    if length != 0:
        ica.apply(raw_copied, exclude=exclude_idx)
    print(f"共排除了{length}个标签")
    return raw_copied


def interpolate_bad_channels(raw):
    """
    绘制脑电图并插值坏导

    :param raw: MNE Raw 数据对象
    """
    raw.plot(block=True, title='请查看坏导')
    user_input = input(
        "请输入要标记为坏导的通道名称，多个通道用空格分隔（例如：MEG 0111 EEG 001）：")  # 不知道为什么交互界面读到的数据是np_str,因此无法交互直接踢除
    bads = [channel.strip() for channel in user_input.split() if channel.strip()]
    raw.info['bads'] = bads
    if len(raw.info['bads']) != 0:
        raw.interpolate_bads()
        print('完成坏导插值')
    else:
        print('没有坏导，不需插值')


def remove_bad_segments(raw, flag, duration=2):
    """
    剔除坏段

    :param raw: MNE Raw 数据对象
    :param flag(bool): 是否需要返回连续的 epochs？True 需要

    :return flag == True：   raw (连续的epochs)
    :return flag == False:  epochs
    """
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, overlap=0.0, preload=True)
    fig = epochs.plot(show=True, title='请选择坏段')
    plt.show(block=True)
    print('完成坏片段剔除')
    raw_list = []
    if flag:
        for epoch in epochs:
            info = epochs.info  # 使用原来的epochs的info
            epoch_raw = mne.io.RawArray(epoch, info)
            raw_list.append(epoch_raw)
        continuous_raw = mne.concatenate_raws(raw_list)
        return continuous_raw
    else:
        return epochs




def preprocess_data(file_dir, save_dir, annotations_df, channels=None, positions=None):
    """
    处理单个患者数据文件，选择指定的通道并进行处理，最后将结果保存为 .npy 文件。n
    在滤波器模块，默认只使用 IIR 滤波器。

    :param file_dir: 原文件目录
    :param save_dir: 保存数据的目录
    :param time_df (DataFrame): 发病时间的 DataFrame。
    :param channels: 需要选择的通道列表，默认为None，使用默认通道列表
    :param positions: 通道位置，默认为None，使用默认通道列表的默认位置
    """

    # 默认电极通道
    if channels == None:
        channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
                    "FZ-CZ", "CZ-PZ", "T7-FT9", "FT9-FT10", "FT10-T8"]
    if positions == None:
        positions = [[-35.0, 58.0, 12.0], [-57.0, 34.0, 17.0], [-63.0, 7.0, 22.0], [-57.0, -22.0, 22.0],
                     [-15.0, 57.0, 10.0], [-35.0, 39.0, 21.0], [-40.0, 9.0, 25.0], [-45.0, -22.0, 24.0],
                     [15.0, 57.0, 10.0], [35.0, 39.0, 21.0], [40.0, 9.0, 25.0], [45.0, -22.0, 24.0],
                     [35.0, 58.0, 12.0], [57.0, 34.0, 17.0], [63.0, 7.0, 22.0], [57.0, -22.0, 22.0],
                     [0.0, 39.0, 22.0], [0.0, 9.0, 25.0], [-75.0, -1.0, 27.5], [0.0, 0.0, 70.0], [75.0, -1.0, 27.5]]


    files = extract_file(file_dir, extension=".edf")
    for file in files:
        file_str = os.path.splitext(file)[0]  # 去除扩展名
        if file_str in annotations_df.index:
            file_path = os.path.join(file_dir, file)
            prev_file_path = None
            if files.index(file) != 0:
                prev_file_index = files.index(file) - 1
                prev_file_path = os.path.join(file_dir, files[prev_file_index])
            print(f'{file_dir}：开始处理 {file}')

            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            print(f'{file_dir}：{file} 读取成功')

            # # 注释
            # seizure_annotations(raw, annotations_df, file_str)
            # print(f'{file_dir}：完成 {file} 注释')

            # 分段，注意留出删除坏片段的时间
            start_time = annotations_df[annotations_df.index == file_str]['start'].values[0]
            duration = 500

            raw_cut = cut_segments(raw=raw, prev_file_path=prev_file_path, start_time=start_time, duration=duration)
            print(f'{file_dir}：{file} 完成第一次片段提取')

            # 预处理，预处理最好放这里，不然内存会爆炸

            print(f'{file_dir}：{file} 开始预处理')
            raw.pick(channels)
            raw_cut.pick(channels)  # 选择指定的通道
            print(f'{file_dir}：{file} 完成通道选择')

            # 设置蒙太奇
            set_montage(raw, channels, positions)
            set_montage(raw_cut, channels, positions)
            print(f'{file_dir}：{file} 完成montage设置')

            # 滤波
            apply_filter(raw_cut)
            print(f'{file_dir}：完成滤波')

            plt.switch_backend('TkAgg')  # You can use this backend if needed
            plt.ion()  # Makes plot interactive
            fig1, (ax1_1, ax1_2) = plt.subplots(2, 1, figsize=(8, 12))
            raw.plot_psd(fmax=100, show=False, ax=ax1_1)  # 原始数据 PSD
            ax1_1.set_title('原始数据')
            raw_cut.plot_psd(fmax=100, show=False, ax=ax1_2)  # 滤波后的数据 PSD
            ax1_2.set_title('滤波后数据')
            plt.tight_layout()  # 显示两张垂直排列的图
            plt.show(block=True)

            # 插值坏导
            interpolate_bad_channels(raw_cut)
            print(f'{file_dir}：{file} 完成第一次坏导检查')

            # 剔除坏段
            epochs = remove_bad_segments(raw_cut, flag=False)

            # 去除伪迹，并将每个epoch连接为Raw
            raw_list = []
            turn = 0
            start = time.time()
            for epoch in epochs:
                info = epochs.info  # 使用原来的epochs的info
                epoch_raw = mne.io.RawArray(epoch, info)
                epoch_cleaned = remove_artefacts(epoch_raw)
                raw_list.append(epoch_cleaned)
                turn += 1
                print(f'-----第 {turn} 个-----\n')
            continuous_raw = mne.concatenate_raws(raw_list)
            continuous_raw.set_annotations(None)
            end = time.time()
            print(f'{file_dir}：{file} 完成伪迹去除, 共用时 {end-start} 秒')

            # 查看去除伪迹之后是否还有残留的伪迹片段（大幅度活动以及眼动）
            fig3 = raw_cut.plot_psd(fmax=100, show=True)
            fig3.suptitle('去伪迹后数据')
            plt.show(block=True)
            print("请检查是否还有残留的坏导和坏片段")
            continuous_raw = remove_bad_segments(continuous_raw, flag=True)
            print(f'{file_dir}：{file} 完成第二次剔除坏片段')

            # 提取目标片段
            start_time1 = continuous_raw.times[-1] - 3
            print(start_time1)
            duration1 = 300
            if start_time1 < duration1:
                raise ValueError("剩余时长不足")
            raw_target = cut_segments(raw=continuous_raw, prev_file_path=None, start_time=start_time1, duration=duration1)
            print(f'{file_dir}：{file} 完成目标片段提取')

            # 绘图对比
            fig3, (ax3_1, ax3_2, ax3_3) = plt.subplots(3, 1, figsize=(10, 12))
            raw.plot_psd(fmax=100, show=False, ax=ax3_1)  # 原始数据 PSD
            ax3_1.set_title('原始数据')
            raw_cut.plot_psd(fmax=100, show=False, ax=ax3_2)  # 滤波后的数据 PSD
            ax3_2.set_title('滤波后数据')
            raw_target.plot_psd(fmax=100, show=False, ax=ax3_3)  # 预处理后的数据 PSD
            ax3_3.set_title('预处理后数据')
            plt.tight_layout()    # 显示三张垂直排列的图
            plt.show(block=True)

            # 保存数据
            epochs = mne.make_fixed_length_epochs(raw_target, reject_by_annotation=False, duration=1, preload=True)  # 窗口0.5s
            epochs_data = epochs.get_data()
            save_path = os.path.join(save_dir, file)
            np.save(save_path.replace('.edf', '.npy'), epochs_data)
            print(f'{file_dir}：{file} 数据保存至 {save_path}')

    print(f'{file_dir}：完成数据预处理')


if __name__ == '__main__':
    # 设置字体为 SimHei
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定 SimHei 字体

    # 解决负号显示问题（可选）
    plt.rcParams['axes.unicode_minus'] = False

    # 存在非法电极(空电极)，这里可以不处理，因为pick channels会选择目标电极 drop 不需要的电极
    invalid_chb = ['chb14', 'chb16', 'chb17', 'chb20', 'chb21', 'chb22']
    empty_channels = ['--0', '--1', '--2', '--3', '--4']

    # 调用函数时传入路径参数
    # kaggle
    time_table_path = "seizures_info.csv"  # 患者发病发病起止时间表，可替换路径
    file_dir = "./CHB-MIT_dataset/chb01"  # 数据文件所在目录
    prev_save_dir = "./CHB-MIT_pre_dataset/preictal"  # 保存的目录

    # 读取发病时间表，结构如下：
    # file（index）     num       start       end       duration
    # chb06_01          1         1724       1738        14
    annotations_df = pd.read_csv(time_table_path, index_col="file")

    if not os.path.exists(prev_save_dir):
        os.makedirs(prev_save_dir)  # 如果目录不存在，创建目录
        print("保存目录不存在，现已建立")
    preprocess_data(file_dir, prev_save_dir, annotations_df)
