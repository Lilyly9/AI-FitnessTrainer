"""
全局 Z-score 标准化 — 替换各数据集独立的 Min-Max 归一化。

问题: 不同数据集使用不同传感器（量程、噪声、安装位置各异），
      各自独立 Min-Max → [0,1] 后，相同物理运动在不同数据集中
      数值范围完全不同，模型难以学习统一特征。

方案: 对合并后的 x_train 计算全局 per-channel mean/std，
      对 train/test 统一做 z-score: (x - mean) / std。
      保留通道间相对关系（重力方向等信息），消除数据集间尺度差异。

用法:
  python src/datasets/normalize_global.py                           # 默认路径
  python src/datasets/normalize_global.py --data_dir data/processed/  # 自定义路径

输出:
  data/processed/x_train.npy  (原地覆盖，同时备份 x_train_minmax.npy)
  data/processed/x_test.npy
  data/processed/norm_params.npz  (mean/std, 供推理时复用)
"""

import numpy as np
import os
import argparse
import shutil

DATA_DIR = 'data/processed/'


def compute_global_norm(x_train):
    """计算全局 per-channel 均值和标准差。

    参数:
      x_train: (N, C, T) 训练集

    返回:
      mean: (C, 1, 1) 每通道均值
      std:  (C, 1, 1) 每通道标准差
    """
    mean = x_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1) → reshape to (C, 1, 1)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)  # 防止除零
    return mean, std


def apply_norm(x, mean, std):
    """应用 z-score 归一化: (x - mean) / std。"""
    return (x.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='全局 Z-score 标准化')
    parser.add_argument('--data_dir', default=DATA_DIR,
                        help=f'数据目录 (默认: {DATA_DIR})')
    args = parser.parse_args()

    data_dir = args.data_dir

    x_train_path = os.path.join(data_dir, 'x_train.npy')
    x_test_path = os.path.join(data_dir, 'x_test.npy')

    if not os.path.exists(x_train_path):
        print(f'错误: 找不到 {x_train_path}')
        return

    # 1) 加载数据
    print('=' * 50)
    print('全局 Z-score 标准化')
    print('=' * 50)
    x_train = np.load(x_train_path).astype(np.float32)
    x_test = np.load(x_test_path).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    print(f'x_train: {x_train.shape}, x_test: {x_test.shape}')

    # 2) 备份原始 Min-Max 数据
    backup_train = os.path.join(data_dir, 'x_train_minmax.npy')
    backup_test = os.path.join(data_dir, 'x_test_minmax.npy')
    if not os.path.exists(backup_train):
        shutil.copy2(x_train_path, backup_train)
        shutil.copy2(x_test_path, backup_test)
        print(f'已备份原始数据到 *_minmax.npy')

    # 3) 计算全局统计量
    mean, std = compute_global_norm(x_train)
    print(f'\n全局统计量 (per-channel):')
    ch_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for c in range(6):
        print(f'  {ch_names[c]:8s}  mean={mean[0,c,0]:+.4f}  std={std[0,c,0]:.4f}')

    # 4) 应用归一化
    x_train_norm = apply_norm(x_train, mean, std)
    x_test_norm = apply_norm(x_test, mean, std)

    print(f'\n归一化后统计:')
    for c in range(6):
        print(f'  {ch_names[c]:8s}  train mean={x_train_norm[:,c,:].mean():+.4f} '
              f'std={x_train_norm[:,c,:].std():.4f}  '
              f'range=[{x_train_norm[:,c,:].min():+.2f}, {x_train_norm[:,c,:].max():+.2f}]')

    # 5) 保存
    np.save(x_train_path, x_train_norm)
    np.save(x_test_path, x_test_norm)
    np.save(y_train_path := os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(y_test_path := os.path.join(data_dir, 'y_test.npy'), y_test)

    # 保存归一化参数（推理时复用）
    np.savez(os.path.join(data_dir, 'norm_params.npz'),
             mean=mean.squeeze(), std=std.squeeze())
    print(f'\n归一化参数已保存到 norm_params.npz')
    print(f'标准化后的数据已保存: x_train.npy, x_test.npy')
    print(f'原始 Min-Max 备份: x_train_minmax.npy, x_test_minmax.npy')
    print('=' * 50)


if __name__ == '__main__':
    main()
