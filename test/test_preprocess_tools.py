import os
import tempfile
import warnings

import pytest
import numpy as np
import pandas as pd
import mne
from mne.annotations import Annotations

from preprocess_data import (
    extract_file,
    seizure_annotations,
    apply_filter,
    cut_segments,
)


@pytest.fixture
def dummy_dir(tmp_path):
    # 创建一个临时目录，里面放几种后缀的文件
    p = tmp_path / "data"
    p.mkdir()
    for fname in ["a.edf", "b.edf", "c.txt", "d"]:
        (p / fname).write_text("dummy")
    return str(p)


def test_extract_file_with_extension_and_keep_ext(dummy_dir):
    files = extract_file(dummy_dir, extension=".edf", need_extension=True)
    # 只返回带 .edf 的文件名
    assert set(files) == {"a.edf", "b.edf"}


def test_extract_file_with_extension_no_ext(dummy_dir):
    files = extract_file(dummy_dir, extension=".edf", need_extension=False)
    # 去掉 .edf 后缀
    assert set(files) == {"a", "b"}


def test_extract_file_no_extension_raises(dummy_dir):
    with pytest.raises(Exception):
        # 不指定 extension，但 need_extension=False 时应报错
        extract_file(dummy_dir, extension=None, need_extension=False)


def make_dummy_raw(duration_s=10.0, sfreq=100.0):
    """生成一个单通道的 RawArray，用于测试"""
    n_samples = int(duration_s * sfreq)
    data = np.zeros((1, n_samples))
    info = mne.create_info(ch_names=["EEG 001"], sfreq=sfreq, ch_types=["eeg"])
    return mne.io.RawArray(data, info)


def test_seizure_annotations_sets_annotations():
    raw = make_dummy_raw(duration_s=20.0, sfreq=100.0)
    # 构造一个 annotations_df
    df = pd.DataFrame([
        {"num": 1, "start": 5.0, "duration": 2.0},
        {"num": 2, "start": 15.0, "duration": 1.0},
    ], index=["file1", "file1"])
    # 应用标注
    seizure_annotations(raw, df, "file1")
    # 检查 Raw.annotations 是否被设置
    assert isinstance(raw.annotations, Annotations)
    # 应该包含两个标注段
    assert len(raw.annotations.onset) == 2
    assert pytest.approx(raw.annotations.duration[0]) == 2.0
    assert pytest.approx(raw.annotations.onset[1]) == 15.0


def test_apply_filter_updates_info():
    raw = make_dummy_raw(duration_s=5.0, sfreq=256.0)
    # 低于 70 Hz 经过低通，高于 0.5 Hz 经过高通
    apply_filter(raw)
    # mne 会把最后一次滤波结果写到 info 里
    # 低通应该是 70 Hz
    assert raw.info["lowpass"] == pytest.approx(70.0)
    # 高通应该是 0.5 Hz
    assert raw.info["highpass"] == pytest.approx(0.5)


def test_cut_segments_simple_before_duration(tmp_path):
    # 创建两个连续的 Raw，prev 和 cur
    sfreq = 100.0
    duration = 2.0  # 2 秒片段长度
    prev_raw = make_dummy_raw(duration_s=3.0, sfreq=sfreq)
    cur_raw = make_dummy_raw(duration_s=3.0, sfreq=sfreq)

    # 先把 prev_raw 存为 edf
    prev_path = tmp_path / "prev.edf"
    mne.io.write_raw_edf(prev_raw, str(prev_path), overwrite=True)

    # start_time 小于 duration，触发拼接逻辑
    seg = cut_segments(cur_raw, str(prev_path), start_time=1.0, duration=duration)
    # 拼接后总时长应约等于 duration
    assert seg.times[-1] == pytest.approx(duration, rel=1e-3)


def test_cut_segments_simple_after_duration():
    raw = make_dummy_raw(duration_s=10.0, sfreq=100.0)
    # start_time 大于 duration，直接裁剪
    seg = cut_segments(raw, prev_file_path="", start_time=5.0, duration=2.5)
    # 片段长度应为 2.5 秒
    assert seg.times[-1] == pytest.approx(2.5, rel=1e-3)


def test_cut_segments_invalid_prev_file(tmp_path):
    raw = make_dummy_raw(duration_s=5.0, sfreq=100.0)
    # prev 文件不存在时应抛出 FileNotFoundError
    with pytest.raises(FileNotFoundError):
        cut_segments(raw, str(tmp_path / "no_such.edf"), start_time=1.0, duration=1.0)
