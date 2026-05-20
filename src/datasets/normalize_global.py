"""
全局 Z-score 标准化 — 在合并后的数据上统一归一化。
修复不同数据集独立 Min-Max 导致的跨数据集特征不可比问题。

与 normalize_global.py (旧版已删除) 的区别:
  - 直接在合并后的 x_train/x_test 上操作，不再需要单数据集文件
  - 备份原始 Min-Max 数据为 *_minmax.npy
  - 保存 norm_params.npz 供推理复用
"""
import numpy as np
import os
import shutil
import argparse

DATA_DIR = 'data/processed/'


def apply_global_zscore(data_dir=DATA_DIR):
    x_train_path = os.path.join(data_dir, 'x_train.npy')
    x_test_path = os.path.join(data_dir, 'x_test.npy')

    x_train = np.load(x_train_path).astype(np.float32)
    x_test = np.load(x_test_path).astype(np.float32)

    print(f"原始数据: train={x_train.shape} test={x_test.shape}")
    print(f"  train range: [{x_train.min():.4f}, {x_train.max():.4f}]")
    print(f"  test range:  [{x_test.min():.4f}, {x_test.max():.4f}]")

    # 备份 Min-Max 数据
    shutil.copy(x_train_path, x_train_path.replace('.npy', '_minmax.npy'))
    shutil.copy(x_test_path, x_test_path.replace('.npy', '_minmax.npy'))
    print("已备份 Min-Max 数据为 *_minmax.npy")

    # 在训练集上计算 per-channel 全局 mean/std
    mean = x_train.mean(axis=(0, 2), keepdims=True)  # (1, 6, 1)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)

    ch_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    print("\nPer-channel 全局统计量 (基于 train):")
    for c in range(6):
        print(f"  {ch_names[c]:12s}  mean={mean[0,c,0]:+.4f}  std={std[0,c,0]:.4f}")

    # 应用 Z-score
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std

    print(f"\nZ-score 后:")
    print(f"  train: mean={x_train_norm.mean():+.4f}  std={x_train_norm.std():.4f}  "
          f"range=[{x_train_norm.min():+.4f}, {x_train_norm.max():+.4f}]")
    print(f"  test:  mean={x_test_norm.mean():+.4f}  std={x_test_norm.std():.4f}  "
          f"range=[{x_test_norm.min():+.4f}, {x_test_norm.max():+.4f}]")

    # 保存
    np.save(x_train_path, x_train_norm.astype(np.float32))
    np.save(x_test_path, x_test_norm.astype(np.float32))

    # 保存归一化参数
    np.savez(os.path.join(data_dir, 'norm_params.npz'),
             mean=mean.squeeze(), std=std.squeeze(),
             channels=ch_names, norm_type='zscore')

    # 更新 domain 文件 (保持一致性)
    for f in ['domain_train.npy', 'domain_test.npy']:
        fp = os.path.join(data_dir, f)
        if os.path.exists(fp):
            print(f"  {f} 保持不变")

    # 检查各数据集 per-class 统计
    meta_path = os.path.join(data_dir, 'dataset_meta.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        y_train = np.load(os.path.join(data_dir, 'y_train.npy')).flatten()
        print("\nPer-dataset Z-score 统计:")
        if 'label_sets' in meta:
            for ds, lbls in meta['label_sets'].items():
                mask = np.isin(y_train, lbls)
                if mask.sum() > 0:
                    ds_data = x_train_norm[mask]
                    print(f"  {ds} (n={mask.sum()}): mean={ds_data.mean():+.4f} "
                          f"std={ds_data.std():.4f} range=[{ds_data.min():+.4f}, {ds_data.max():+.4f}]")

    print("\n[OK] 全局 Z-score 标准化完成!")
    print(f"恢复 Min-Max 版本: 将 *_minmax.npy 改回 *.npy 即可")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    args = parser.parse_args()
    apply_global_zscore(args.data_dir)
