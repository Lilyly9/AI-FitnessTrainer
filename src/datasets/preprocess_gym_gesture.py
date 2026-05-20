"""
Gym Gesture 数据集预处理（已有，重构为模块化版本）
================================================================
来源: https://ieee-dataport.org/documents/gym-gesture-classification-using-imu-sensor-dataset
规模: 5 人 × 5 动作，150,000 条
传感器: Arduino Nano 33 BLE, 100Hz, 手腕
动作: chest_fly, chest_press, lat_pulldown, seated_row, tricep_extension

使用方法:
  python src/datasets/preprocess_gym_gesture.py

输出: data/processed/gym_gesture_x_train.npy, gym_gesture_y_train.npy, ...
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

DATA_PATH = 'data/raw/gym_gesture/imu_dataset.csv'
OUT_DIR = 'data/processed/'
WINDOW_SIZE = 200
STEP_SIZE = 100
SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABEL_COL = 'label'
ID_COL = 'athlete_id'
LABEL_MAP = {
    'chest_fly': 0, 'chest_press': 1, 'lat_pulldown': 2,
    'seated_row': 3, 'tricep_extension': 4
}
TRAIN_IDS = [1, 2, 3, 4]
TEST_IDS = [5]
DOMAIN_ID = 'wrist'  # Gym Gesture 传感器佩戴位置：手腕 (Arduino Nano 33 BLE)


def create_windows(group, window_size, step_size, sensor_cols, label_col):
    data = group[sensor_cols].values
    labels = group[label_col].values
    windows, window_labels = [], []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end].T
        if np.isnan(window).any():
            continue
        windows.append(window)
        window_labels.append(Counter(labels[start:end]).most_common(1)[0][0])
    return np.array(windows), np.array(window_labels)


def process_dataset(df, window_size, step_size, sensor_cols, id_col, label_col):
    all_windows, all_labels = [], []
    for athlete_id, group in df.groupby(id_col):
        group = group.reset_index(drop=True)
        wins, labs = create_windows(group, window_size, step_size, sensor_cols, label_col)
        all_windows.append(wins)
        all_labels.append(labs)
        print(f"  受试者 {athlete_id}: {len(wins)} 个窗口")
    return np.concatenate(all_windows), np.concatenate(all_labels)


def preprocess():
    print("=" * 50)
    print("Gym Gesture 数据集预处理")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        print(f"错误: 未找到 {DATA_PATH}")
        print("请从 IEEE DataPort 下载数据集:")
        print("  https://ieee-dataport.org/documents/gym-gesture-classification-using-imu-sensor-dataset")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape}")

    # 列名标准化
    COLUMN_RENAME = {
        'ax': 'acc_x', 'ay': 'acc_y', 'az': 'acc_z',
        'gx': 'gyro_x', 'gy': 'gyro_y', 'gz': 'gyro_z',
        'exercise_type': 'label',
    }
    df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns}, inplace=True)

    # 标签映射
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].map(LABEL_MAP)

    # 缺失值 & 提取列
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill()
    df = df[SENSOR_COLS + [LABEL_COL, ID_COL]].copy()

    # 按受试者划分
    train_df = df[df[ID_COL].isin(TRAIN_IDS)].copy()
    test_df = df[df[ID_COL].isin(TEST_IDS)].copy()

    # Min-Max 归一化（基于训练集）
    min_v, max_v = train_df[SENSOR_COLS].min(), train_df[SENSOR_COLS].max()
    denom = max_v - min_v
    denom[denom < 1e-8] = 1.0
    train_df[SENSOR_COLS] = (train_df[SENSOR_COLS] - min_v) / denom
    test_df[SENSOR_COLS] = (test_df[SENSOR_COLS] - min_v) / denom

    # 滑动窗口
    print("处理训练集...")
    x_train, y_train = process_dataset(train_df, WINDOW_SIZE, STEP_SIZE, SENSOR_COLS, ID_COL, LABEL_COL)
    print("处理测试集...")
    x_test, y_test = process_dataset(test_df, WINDOW_SIZE, STEP_SIZE, SENSOR_COLS, ID_COL, LABEL_COL)

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)
    dataset_name = 'gym_gesture'
    # 带前缀版本（供合并脚本使用）
    np.save(os.path.join(OUT_DIR, f'{dataset_name}_x_train.npy'), x_train)
    np.save(os.path.join(OUT_DIR, f'{dataset_name}_y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, f'{dataset_name}_x_test.npy'), x_test)
    np.save(os.path.join(OUT_DIR, f'{dataset_name}_y_test.npy'), y_test)
    # 不带前缀版本（向后兼容，train.py 直接用）
    np.save(os.path.join(OUT_DIR, 'x_train.npy'), x_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

    # 保存标签映射 {label_id: exercise_name}
    mapping_df = pd.DataFrame([
        {'label': v, 'name': k} for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])
    ])
    mapping_df.to_csv(os.path.join(OUT_DIR, f'{dataset_name}_label_mapping.csv'), index=False)

    # Domain label: wrist (domain_id=1)
    domain_train = np.ones(len(x_train), dtype=np.int64)
    domain_test = np.ones(len(x_test), dtype=np.int64)
    np.save(os.path.join(OUT_DIR, f'{dataset_name}_domain_train.npy'), domain_train)
    np.save(os.path.join(OUT_DIR, f'{dataset_name}_domain_test.npy'), domain_test)
    print(f"Domain label: {DOMAIN_ID} (id=1)")
    print(f"已保存到 data/processed/")


if __name__ == '__main__':
    preprocess()
