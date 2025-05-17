# 预处理 CHB-MIT 数据集

本项目旨在对 [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) 癫痫EEG数据集进行结构化预处理，提供从数据切割、伪迹剔除到标准格式保存的完整流程，便于后续机器学习模型的训练与分析。


---

## 🧠数据集

本项目预处理流程基于 CHB-MIT Scalp EEG Database，该数据库由儿童医院波士顿和麻省理工学院合作采集，记录了 22 名小儿癫痫患者的长期脑电数据，共 664 个 EEG 记录，其中 198 次包含发作事件。  
数据可从 PhysioNet 获取：<https://physionet.org/content/chbmit/1.0.0/>。  
该数据库 DOI 为 `10.13026/C2K01R`，RRID: `SCR_004264`。

**标准引用格式**  
> Shoeb, A. H. *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, Massachusetts Institute of Technology, September 2009.  
> Goldberger, A. L., et al. “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.” *Circulation* **101**(23):e215–e220, June 13, 2000.
---




## 🚀 快速开始

### 1. 安装依赖
在安装前可运行以下命令更新 pip

```bash
python -m pip install --upgrade pip
```

推荐使用 Python 3.10 或以上版本，建议创建虚拟环境。

运行以下命令启用可编辑模式安装（建议使用😀）

```bash
pip install -e .
```

或从 `pyproject.toml` 安装（需 `pip >= 25.1.1`）：

```bash
pip install .
```
❗ `mne_icalabel` 依赖项需手动安装，可以访问 MNE-ICALabel 下载入口获取相关信息及下载指导：

👉 https://mne.tools/mne-icalabel/stable/install.html

### 2. 修改项目设置（参考）
可以参考以下两点
- 修改切割片段的终止时间（即 `cut_segment` 函数的 `start` 参数）
- 设计你自己的滤波器

### 3. 设置 `config.toml`

配置 `config.toml` 的关键说明：

❗首先，请务必先将选择的患者的发病数据汇总为 `.csv` 文件，文件格形式如下（字母与逗号以及数字与逗号间无空格）：
```csv
file, num, start, end, duration
chb06_01, 1, 1724, 1738, 14
chb06_01, 2, 1800, 1838, 38
```
表头分别为记录的片段文件名（file）、第几次发作（num）、发作开始时间（start）、发作结束时间（end）和发作持续时长（duration）。
这一步强烈建议借助AI完成。

其次，将 `data_dir` 、 `output_dir` 和 `annotations_path` 分别更改为你的数据集目录（包含多个子文件夹）、保存目录和发作信息 `.csv` 文件。

最后，可以修改其他参数，使其更符合个人应用场景。

### 4. 运行主程序
直接在终端运行以下指令（需要使用 `pip install -e .` 或打包安装）
```bash
run-pre
```
当然，你可以直接在 `pyproject.toml` 配置中将该指令修改为你喜欢的方式，别忘了在更改后再运行一遍`pip install -e .` 或打包安装哦。

没有使用 `pip install -e .` 或打包安装，可运行以下指令
```bash
python -m preprocessing_CHB_MIT_dataset_project.main
```

还可以通过 CLI 指定配置文件：

```bash
python -m preprocessing_CHB_MIT_dataset_project.main --config config.toml
```

---

## ⚙️ 配置文件示例：`config.toml`

示例：

```toml
[paths]
data_dir   = "your/own/dataset/dir/"
output_dir = "dir/to/save/output/.npy/data/"
annotations_path = "your/own/annotations/csv/path"

[preprocessing]
picked_channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                     "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                     "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                     "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
                     "FZ-CZ", "CZ-PZ", "T7-FT9", "FT9-FT10", "FT10-T8"]
channel_positions = [[-35.0, 58.0, 12.0], [-57.0, 34.0, 17.0], [-63.0, 7.0, 22.0], [-57.0, -22.0, 22.0],
                     [-15.0, 57.0, 10.0], [-35.0, 39.0, 21.0], [-40.0, 9.0, 25.0], [-45.0, -22.0, 24.0],
                     [15.0, 57.0, 10.0], [35.0, 39.0, 21.0], [40.0, 9.0, 25.0], [45.0, -22.0, 24.0],
                     [35.0, 58.0, 12.0], [57.0, 34.0, 17.0], [63.0, 7.0, 22.0], [57.0, -22.0, 22.0],
                     [0.0, 39.0, 22.0], [0.0, 9.0, 25.0], [-75.0, -1.0, 27.5], [0.0, 0.0, 70.0], [75.0, -1.0, 27.5]]
# 裁剪设置
first_segment_duration = 30  #首次裁剪的片段长度（示例为发病前30秒）
second_segment_duration = 20 #最终确定保留的片段长度
# ICA设置
artifact_epoch_duration = 2
ICA_components = 16
# 导出的每个epoch的长度
output_epoch_duration = 1
overlap_present = 0  # 每个epoch间多少重叠
# 自动剔除坏段坏导设置（建议在癫痫数据集开启）
interactive = true
# 并行处理，当不需要交互时开启
parallel = false
```

---

## 📁 项目结构

```
preprocessing_CHB_MIT_dataset_project/
    │
    ├── pyproject.toml          # 项目元数据和依赖管理
    │
    ├── config.toml             # 自定义配置参数
    │
    ├── README.md               # 项目说明文档
    │
    ├── src/
    │   │
    │   └── preprocessing_CHB_MIT_dataset/
    │           │
    │           ├── __init__.py
    │           │
    │           ├── main.py              # 主程序入口
    │           │
    │           ├── preprocess_data.py   # 数据预处理模块
    │           │
    │           └── utils.py             # 工具函数
    │
    └── dataset/                 # CHB-MIT数据集
        │
        ├──chb01
        │   │
        │   ├──chb01_01.edf
        :   :
```

---

## 📦 模块功能说明

- `main.py`：主入口程序，读取配置文件并启动整个预处理流程。
- `preprocess.py`：核心数据处理模块，按患者数据文件逐个执行伪迹剔除、通道插值、滤波、分段等操作。
- `utils.py`：提供各种辅助函数，如通道位置设定、ICA 伪迹识别、自动标注癫痫发作时间段等。

---

## 📄 License

MIT License. 你可以自由使用和修改本项目。

---

## ✍️ 作者

- Yang Ke（2418423476@qq.com）
- 若为科研用途，请在引用中注明本项目名称与作者

# preprocessing-CHB-MIT-dataset

This project aims to perform structured preprocessing on the [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) epileptic EEG dataset, providing a complete pipeline from data segmentation, artifact removal, to standardized format saving, facilitating subsequent machine learning model training and analysis.

---

## 🧠 Dataset

The preprocessing workflow is based on the CHB-MIT Scalp EEG Database, jointly collected by Boston Children's Hospital and MIT. It contains long-term EEG recordings from 22 pediatric epilepsy patients, comprising 664 EEG records with 198 seizure events.  
Data is available on PhysioNet: <https://physionet.org/content/chbmit/1.0.0/>.  
Database DOI: `10.13026/C2K01R`, RRID: `SCR_004264`.

**Standard Citation**  
> Shoeb, A. H. *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, Massachusetts Institute of Technology, September 2009.  
> Goldberger, A. L., et al. “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.” *Circulation* **101**(23):e215–e220, June 13, 2000.

---

## 🚀 Quick Start

### 1. Install Dependencies
Update pip before installation:
```bash
python -m pip install --upgrade pip
```

Python 3.10 or higher is recommended. Use a virtual environment for best practices.

Install in editable mode (recommended 😀):
```bash
pip install -e .
```

Or install from `pyproject.toml` (requires `pip >= 25.1.1`):
```bash
pip install .
```
❗ Note: The `mne_icalabel` dependencies require manual installation. For detailed instructions and download guidance, please visit the MNE-ICALabel download/installation guide:

👉 https://mne.tools/mne-icalabel/stable/install.html

### 2. Modify Project Settings (Reference)
Key customizable parameters:
- Adjust the termination time for data segmentation (i.e., the `start` parameter in the `cut_segment` function).
- Design custom filters.

### 3. Configure `config.toml`
Critical configurations in `config.toml`:

❗ First, compile seizure event information into a `.csv` template as following  (space between English latter and comma or number and comma is not expected):
```csv
file, num, start, end, duration
chb06_01, 1, 1724, 1738, 14
chb06_01, 2, 1800, 1838, 38
```
Headers represent: record filename (`file`), seizure occurrence index (`num`), seizure start time (`start`), end time (`end`), and duration (`duration`). AI tools are recommended for generating this file.

Second, update `data_dir`, `output_dir`, and `annotations_path` to your dataset directory (containing subfolders), output directory, and seizure annotation `.csv` file path.

Finally, modify other parameters to suit your application.

### 4. Run Main Program
Execute via terminal (requires `pip install -e .` or package installation):
```bash
run-pre
```
You may modify the command in `pyproject.toml` and reinstall the package.

Without package installation, run:
```bash
python -m preprocessing_CHB_MIT_dataset_project.main
```

Specify a custom configuration file via CLI:
```bash
python -m preprocessing_CHB_MIT_dataset_project.main --config config.toml
```

---

## ⚙️ Example Configuration: `config.toml`

```toml
[paths]
data_dir   = "your/own/dataset/dir/"
output_dir = "dir/to/save/output/.npy/data/"
annotations_path = "your/own/annotations/csv/path"

[preprocessing]
picked_channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                     "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                     "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                     "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
                     "FZ-CZ", "CZ-PZ", "T7-FT9", "FT9-FT10", "FT10-T8"]
channel_positions = [[-35.0, 58.0, 12.0], [-57.0, 34.0, 17.0], [-63.0, 7.0, 22.0], [-57.0, -22.0, 22.0],
                     [-15.0, 57.0, 10.0], [-35.0, 39.0, 21.0], [-40.0, 9.0, 25.0], [-45.0, -22.0, 24.0],
                     [15.0, 57.0, 10.0], [35.0, 39.0, 21.0], [40.0, 9.0, 25.0], [45.0, -22.0, 24.0],
                     [35.0, 58.0, 12.0], [57.0, 34.0, 17.0], [63.0, 7.0, 22.0], [57.0, -22.0, 22.0],
                     [0.0, 39.0, 22.0], [0.0, 9.0, 25.0], [-75.0, -1.0, 27.5], [0.0, 0.0, 70.0], [75.0, -1.0, 27.5]]
# Segmentation settings
first_segment_duration = 30  # Initial segment length (e.g., 30s pre-seizure)
second_segment_duration = 20 # Final retained segment length
# ICA settings
artifact_epoch_duration = 2
ICA_components = 16
# Epoch export settings
output_epoch_duration = 1
overlap_present = 0  # Overlap between epochs
# Automated artifact rejection (recommended for epilepsy data)
interactive = true
# Parallel processing (enable when no interaction needed)
parallel = false
```

---

## 📁 Project Structure

```
preprocessing_CHB_MIT_dataset_project/
    │
    ├── pyproject.toml          # Project metadata & dependencies
    │
    ├── config.toml             # Custom configurations
    │
    ├── README.md               # Documentation
    │
    ├── src/
    │   │
    │   └── preprocessing_CHB_MIT_dataset/
    │           │
    │           ├── __init__.py
    │           │
    │           ├── main.py              # Main entry point
    │           │
    │           ├── preprocess_data.py   # Core preprocessing module
    │           │
    │           └── utils.py             # Utility functions
    │
    └── dataset/                 # CHB-MIT dataset
        │
        ├──chb01
        │   │
        │   ├──chb01_01.edf
        :   :
```

---

## 📦 Module Descriptions

- `main.py`: Main entry point; reads configurations and initiates preprocessing.
- `preprocess_data.py`: Core module for artifact removal, channel interpolation, filtering, and segmentation.
- `utils.py`: Helper functions (e.g., channel positioning, ICA artifact detection, seizure annotation).

---

## 📄 License

MIT License. Free to use and modify.

---

## ✍️ Authors

- Yang Ke（2418423476@qq.com）
- Please cite this project and authors in academic use.