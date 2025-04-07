'''
Copyright (c) 2025 Yang Ke. All rights reserved.
Python Version: 3.10
Project Name: main.py
Project Version: v1.0.2
Author: Yang Ke
Created: 2025/4/7
Project: main.py
File: mm.py
GitHub: https: https://github.com/UKON09/preprocessing-CHB-MIT-dataset
'''

import mne
from mne.annotations import Annotations
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pywt
import warnings
import time


def extract_file(file_dir, extension=None, need_extension=True):
    """
    获取目标路径下的文件名

    参数:
    - file_dir：目标文件所在的路径
    - extension：目标文件后缀，例如名为'xx.edf'的'.edf' (默认为 None)
    - need_extension: 是否需要保留名字中的后缀 (默认为 True)

    返回:
    - files(list): 包含文件名的列表
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

    参数:
    raw: .edf 数据文件
    annotations_df(DataFrame): 注释信息
    num: 一个文件中发作的次数
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

    参数:
    - raw: MNE Raw 数据对象
    """
    # 带通通滤波器
    raw.filter(l_freq=0.1, h_freq=85, method='iir', iir_params=dict(order=4, ftype='butter'))
    # 60Hz工频干扰
    raw.filter(l_freq=61, h_freq=58, method='fir', fir_design='firwin', phase='zero-double', filter_length='auto',
               l_trans_bandwidth=1, h_trans_bandwidth=1)
    # 50Hz
    raw.filter(l_freq=51, h_freq=47, method='iir', iir_params=dict(order=2, ftype='butter'))


def cut_segments(raw, prev_file_path, start_time, duration):
    """
    根据注释的开始时间和持续时间来切割数据。
    若注释的开始时间 > 3600 秒，则从 (start_time - 3600, start_time) 进行切割。
    若注释的开始时间 <= 3600 秒，则先将该片段与上一片段连接，再裁剪 (start_time - 3600, start_time)。

    参数:
    raw (mne.io.Raw): 原始数据对象
    prev_file_path: 前一个.edf文件地址

    返回:
    segments (mne.io.Raw): 切割后的数据片段
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


def set_montage(raw, channels):
    '''
    由于电极名称设置不规范，所以将大部分电极位映射到 1020标准电极位， T7-FT9 和 FT10-T8 单独计算，后 montage
    以下是映射信息，其中 001 为 T7、FT9 电极空间坐标的中值，002 为 FT10、T8 电极空间坐标的中值
    new_channels = ['AF7', 'FT7', 'TP7', 'PO7',
                    'AF3', 'FC3', 'CP3', 'PO3',
                    'AF4', 'FC4', 'CP4', 'PO4',
                    'AF8', 'FT8', 'TP8', 'PO8',
                    'FCz', 'CPz', '001', 'Nz', '002']

    参数：
    - raw_data: MNE Raw 数据对象
    - channels: 双极导联电极名称表
    '''

    positions = [[-35.0, 58.0, 12.0], [-57.0, 34.0, 17.0], [-63.0, 7.0, 22.0], [-57.0, -22.0, 22.0],
                 [-15.0, 57.0, 10.0], [-35.0, 39.0, 21.0], [-40.0, 9.0, 25.0], [-45.0, -22.0, 24.0],
                 [15.0, 57.0, 10.0], [35.0, 39.0, 21.0], [40.0, 9.0, 25.0], [45.0, -22.0, 24.0],
                 [35.0, 58.0, 12.0], [57.0, 34.0, 17.0], [63.0, 7.0, 22.0], [57.0, -22.0, 22.0],
                 [0.0, 39.0, 22.0], [0.0, 9.0, 25.0], [-75.0, -1.0, 27.5], [0.0, 0.0, 70.0], [75.0, -1.0, 27.5]]
    new_positions = [[num / 2000 for num in sub_list] for sub_list in positions]
    channels_positions = dict(zip(channels, new_positions))
    montage = mne.channels.make_dig_montage(ch_pos=channels_positions, coord_frame='head')
    raw.set_montage(montage)


def remove_artefacts(raw, n_components=15, threshold=0.9):
    """
    使用 ICA 和 ICLabel 自动去除伪迹成分（如眼动、心电图伪迹）。

    参数：
    - raw: 原始EEG数据（Raw对象）
    - n_components: ICA分解的成分数（默认为20）
    - threshold: 用于判定伪迹成分的概率阈值（默认为0.5）

    返回：
    - raw_clean: 去除伪迹后的EEG数据（Raw对象）
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

    # #手动排查
    # ica.plot_components()
    # plt.show(block=True)
    # user_input = input("是否需要查看成分的具体情况？请输入序号查看")
    # while user_input != '':
    #     numbers = int(user_input)
    #     # 查看所有成分的拓扑图
    #     ica.plot_components()
    #     ica.plot_properties(raw_copied, picks=[numbers])
    #     plt.show(block=True)
    #     exc_num = input('是否排除？')
    #     if exc_num != '':
    #         exclude_idx.append(exc_num)
    #     ica.plot_components()
    #     plt.show(block=True)
    #     user_input = input("是否继续查看成分的具体情况？请输入序号查看，按回车退出")

    length = len(exclude_idx)
    if length != 0:
        ica.apply(raw_copied, exclude=exclude_idx)
    print(f"共排除了{length}个标签")
    return raw_copied


def wavelet_denoise(signal, wavelet='db4', level=5, threshold_scale=1.0):
    """
    使用小波变换去噪
    :param signal: 输入信号（1D数组）
    :param wavelet: 小波基类型（如'db4'）
    :param level: 分解层数
    :param threshold_scale: 阈值缩放因子（控制去噪强度）
    :return: 去噪后的信号
    """
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 计算阈值（通用阈值法）
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745  # 噪声标准差估计
    threshold = sigma * np.sqrt(2 * np.log(len(signal))) * threshold_scale

    # 阈值处理（软阈值）
    coeffs_denoised = []
    for i in range(1, len(coeffs)):
        coeffs_denoised.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    # 重构信号
    return pywt.waverec([coeffs[0]] + coeffs_denoised, wavelet)


def preprocess_data(file_dir, save_dir, annotations_df, channels=None):
    """
    处理单个患者数据文件，选择指定的通道并进行处理，最后将结果保存为 .npy 文件。n
    在滤波器模块，默认只使用 IIR 滤波器。

    参数:
    - file_dir: 原文件目录
    - save_dir: 保存数据的目录
    - time_df (DataFrame): 发病时间的 DataFrame。
    - channels: 需要选择的通道列表，默认为None，使用默认通道列表
    """

    # 默认电极通道
    if channels == None:
        channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
                    "FZ-CZ", "CZ-PZ", "T7-FT9", "FT9-FT10", "FT10-T8"]


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

            # 注释
            seizure_annotations(raw, annotations_df, file_str)
            print(f'{file_dir}：完成 {file} 第一次注释')

            #分段，注意留出删除坏片段的时间
            start_time = 0
            duration = 3900
            filtered_df = annotations_df[annotations_df.index == file_str]
            num = len(filtered_df)
            for i in range(1, num + 1):
                start_time = filtered_df[(filtered_df['num'] == i)]['start'].values[0]

            raw_cut = cut_segments(raw, prev_file_path, start_time, duration)
            print(f'{file_dir}：{file} 完成第一次片段提取')

            # 预处理，预处理最好放这里，不然内存会爆炸

            print(f'{file_dir}：{file} 开始预处理')
            raw_cut.pick(channels)  # 选择指定的通道
            print(f'{file_dir}：{file} 完成通道选择')

            # 设置蒙太奇
            set_montage(raw_cut, channels)
            print(f'{file_dir}：{file} 完成montage设置')

            # 滤波
            apply_filter(raw_cut)
            print(f'{file_dir}：完成滤波')
            raw_cut.plot_psd(fmax=110)
            plt.show(block=True)

            # 插值坏导
            raw_cut.plot(block=True, title='请选择坏导')
            user_input = input("请输入要标记为坏导的通道名称，多个通道用空格分隔（例如：MEG 0111 EEG 001）：") # 不知道为什么交互界面读到的数据是np_str,因此无法交互直接踢除
            bads = [channel.strip() for channel in user_input.split() if channel.strip()]
            raw_cut.info['bads'] = bads
            if len(raw_cut.info['bads']) != 0:
                raw_cut.interpolate_bads()
                print(f'{file_dir}：{file} 完成坏导插值')
            else:
                print(f'{file_dir}：{file} 没有坏导，不需插值')

            # 去除伪迹
            epochs = mne.make_fixed_length_epochs(raw_cut, duration=10.0, overlap=0.0, preload=True)
            # 将每个epoch转换为Raw格式并存储在列表中
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
            end = time.time()
            print(f'{file_dir}：{file} 完成伪迹去除, 共用时 {end-start} 秒')

            # 剔除坏段
            epochs = mne.make_fixed_length_epochs(raw_cut, duration=10.0, overlap=0.0, preload=True)
            fig = epochs.plot(show=True, title='请选择坏段')
            plt.show(block=True)
            # 将每个epoch转换为Raw格式并存储在列表中
            raw_list = []
            for epoch in epochs:
                info = epochs.info  # 使用原来的epochs的info
                epoch_raw = mne.io.RawArray(epoch, info)
                raw_list.append(epoch_raw)
            continuous_raw = mne.concatenate_raws(raw_list)
            print(f'{file_dir}：{file} 完成剔除坏片段')


            # 如果在上述过程发现了坏导
            user_input = input(
                "请输入要标记为坏导的通道名称，多个通道用空格分隔（例如：MEG 0111 EEG 001）：")  # 不知道为什么交互界面读到的数据是np_str,因此无法交互直接踢除
            bads2 = [channel.strip() for channel in user_input.split() if channel.strip()]
            continuous_raw.info['bads'] = bads2
            if len(continuous_raw.info['bads']) != 0:
                continuous_raw.interpolate_bads()
                print(f'{file_dir}：{file} 完成坏导插值')
            else:
                print(f'{file_dir}：{file} 没有坏导，不需插值')

            # 提取目标片段
            start_time = continuous_raw.times[-1]
            duration = 3600
            if start_time < duration:
                raise ValueError("剩余时长不足一小时，最多删减50个epoch！！")
            raw_target = cut_segments(continuous_raw, prev_file_path, start_time, duration)
            print(f'{file_dir}：{file} 完成目标片段提取')

            #绘图对比
            raw_target.plot_psd(fmax=110)
            plt.show(block=True)
            raw_target.plot(block=True, title='目标片段')

            # 保存数据
            epochs = mne.make_fixed_length_epochs(raw_target, duration=5, preload=True) # 分段每5秒分割
            save_path = os.path.join(save_dir, file)
            np.save(save_path.replace('.edf', '.npy'), epochs)
            print(f'{file_dir}：{file} 数据保存至 {save_path}')

    print(f'{file_dir}：完成数据预处理')


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    plt.ion()
    # 存在非法电极(空电极)，这里可以不处理，因为pick channels会选择目标电极 drop 不需要的电极
    invalid_chb = ['chb14', 'chb16', 'chb17', 'chb20', 'chb21', 'chb22']
    empty_channels = ['--0', '--1', '--2', '--3', '--4']

    # 调用函数时传入路径参数
    time_table_path = "seizures_info.csv"  # 患者发病发病起止时间表
    file_dir = "./CHB-MIT_dataset/chb01"  # 数据文件所在目录
    save_dir = r"E:\data_preprocessing\CHB-MIT\CHB-MIT_pre_dataset"  # 保存的目录

    # 读取发病时间表，结构如下：
    # file（index）     num       start       end       duration
    # chb06_01          1         1724       1738        14
    annotations_df = pd.read_csv(time_table_path, index_col="file")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果目录不存在，创建目录
        print("保存目录不存在，现已建立")
    preprocess_data(file_dir, save_dir, annotations_df)
