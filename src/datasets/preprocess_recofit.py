"""
Microsoft RecoFit 数据集预处理
================================================================
来源: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
规模: 200+ 人，多种健身房动作
传感器: 加速度计 + 陀螺仪 (arm-band)
格式: .mat (MATLAB) → 统一 .npy

使用方法:
  1. 下载数据集放到 data/raw/recofit/
  2. python src/datasets/preprocess_recofit.py

输出: data/processed/recofit_x_train.npy, recofit_y_train.npy, ...
"""

import numpy as np
import os
from collections import Counter

DATA_DIR = 'data/raw/recofit/'
OUT_DIR = 'data/processed/'
WINDOW_SIZE = 200
STEP_SIZE = 100

# RecoFit 动作标签映射（需要根据实际数据调整）
# 原文中动作名示例，实际以 .mat 文件中的标签为准
EXERCISE_NAMES = {
    # 常见 RecoFit 动作名 → 数字标签
    # 需根据实际数据集调整
}


def load_mat(filepath):
    """加载 .mat 文件（需要 scipy）。"""
    try:
        from scipy.io import loadmat
        return loadmat(filepath)
    except ImportError:
        raise ImportError("需要 scipy: pip install scipy")


def create_windows(data, labels, window_size, step_size):
    windows, window_labels = [], []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end].T  # (channels, time)
        if np.isnan(window).any():
            continue
        windows.append(window)
        window_labels.append(Counter(labels[start:end]).most_common(1)[0][0])
    return np.array(windows), np.array(window_labels)


def preprocess():
    print("=" * 50)
    print("Microsoft RecoFit 数据集预处理")
    print("=" * 50)

    # 扫描数据文件
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    if not mat_files:
        print(f"错误: {DATA_DIR} 中没有 .mat 文件")
        print("请先从以下地址下载数据集:")
        print("  https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors")
        print("  并解压到 data/raw/recofit/")
        return

    print(f"发现 {len(mat_files)} 个 .mat 文件")
    print()
    print("注意: 此脚本为框架代码。RecoFit 的 .mat 文件结构因版本而异，")
    print("请根据实际文件中的数据字段名调整 load_mat() 和后续处理逻辑。")
    print("参考字段: 'data' (传感器), 'labels' (标签), 'subject' (受试者)")
    print()
    print("预处理框架已就绪，适配完成后运行即可。")
    print("=" * 50)


if __name__ == '__main__':
    preprocess()
