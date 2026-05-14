"""
RecGym 数据集预处理
================================================================
来源: https://archive.ics.uci.edu/dataset/1128
规模: 10 人 × 5 次 × 12 类动作，约 440 万条
传感器: IMU (加速度计+陀螺仪) + 身体电容，3个佩戴位置 (手腕/口袋/小腿)
采样率: 20Hz
格式: CSVs → 统一 .npy

动作类别 (12类):
  Adductor, ArmCurl, BenchPress, LegCurl, LegPress, Riding,
  RopeSkipping, Running, Squat, StairsClimber, Walking, Null

使用方法:
  1. 下载数据集解压到 data/raw/recgym/
  2. python src/datasets/preprocess_recgym.py

输出: data/processed/recgym_x_train.npy, recgym_y_train.npy, ...
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

DATA_DIR = 'data/raw/recgym/'
OUT_DIR = 'data/processed/'
WINDOW_SIZE = 200
STEP_SIZE = 100
SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# RecGym 12类动作
LABEL_MAP = {
    'Adductor': 0,
    'ArmCurl': 1,
    'BenchPress': 2,
    'LegCurl': 3,
    'LegPress': 4,
    'Riding': 5,
    'RopeSkipping': 6,
    'Running': 7,
    'Squat': 8,
    'StairsClimber': 9,
    'Walking': 10,
    'Null': 11,
}


def create_windows(data, labels, window_size, step_size):
    windows, window_labels = [], []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size].T
        if np.isnan(window).any():
            continue
        windows.append(window)
        window_labels.append(Counter(labels[start:start + window_size]).most_common(1)[0][0])
    return np.array(windows), np.array(window_labels)


def preprocess(sensor_position='wrist'):
    print("=" * 50)
    print(f"RecGym 数据集预处理 (传感器位置: {sensor_position})")
    print("=" * 50)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误: {DATA_DIR} 中没有 CSV 文件")
        print("请先从以下地址下载数据集:")
        print("  https://archive.ics.uci.edu/dataset/1128")
        print("  并解压到 data/raw/recgym/")
        return

    print(f"发现 {len(csv_files)} 个 CSV 文件")
    print(f"动作类别: {list(LABEL_MAP.keys())}")
    print()
    print("注意: 此脚本为框架代码。RecGym 的具体 CSV 列名和文件组织结构")
    print("请根据实际文件调整 read_csv 参数和列名映射。")
    print()
    print("预处理框架已就绪，适配完成后运行即可。")
    print("=" * 50)


if __name__ == '__main__':
    preprocess()
