"""
多数据集合并脚本 — 支持子文件夹和带前缀两种路径格式。

用法:
  python src/datasets/merge_datasets.py --datasets gym_gesture recofit mmfit
"""

import numpy as np
import pandas as pd
import os
import argparse
import json

OUT_DIR = 'data/processed/'


def find_dataset_files(name):
    """查找数据集文件，支持两种路径格式：
    1. data/processed/{name}/{x_train,y_train,x_test,y_test}.npy  (子文件夹)
    2. data/processed/{name}_{x_train,y_train,x_test,y_test}.npy (前缀)
    """
    patterns = [
        # 子文件夹格式
        (os.path.join(OUT_DIR, name, 'x_train.npy'),
         os.path.join(OUT_DIR, name, 'y_train.npy'),
         os.path.join(OUT_DIR, name, 'x_test.npy'),
         os.path.join(OUT_DIR, name, 'y_test.npy')),
        # 前缀格式
        (os.path.join(OUT_DIR, f'{name}_x_train.npy'),
         os.path.join(OUT_DIR, f'{name}_y_train.npy'),
         os.path.join(OUT_DIR, f'{name}_x_test.npy'),
         os.path.join(OUT_DIR, f'{name}_y_test.npy')),
    ]
    for tx, ty, ex, ey in patterns:
        if os.path.exists(tx):
            return tx, ty, ex, ey
    return None, None, None, None


def merge_datasets(dataset_names):
    x_train_parts, y_train_parts = [], []
    x_test_parts, y_test_parts = [], []
    label_offsets = {}
    all_label_names = []
    offset = 0

    for name in dataset_names:
        tx, ty, ex, ey = find_dataset_files(name)
        if tx is None:
            print(f"跳过 {name}: 未找到预处理文件")
            continue

        x_tr = np.load(tx)
        y_tr = np.load(ty)
        x_te = np.load(ex)
        y_te = np.load(ey)

        # 加载标签映射
        mapping_path = os.path.join(OUT_DIR, name, 'label_mapping.csv')
        if not os.path.exists(mapping_path):
            mapping_path = os.path.join(OUT_DIR, f'{name}_label_mapping.csv')
        if os.path.exists(mapping_path):
            mapping = pd.read_csv(mapping_path)
            for _, row in mapping.iterrows():
                all_label_names.append({
                    'dataset': name,
                    'local_label': int(row['label']),
                    'global_label': int(row['label']) + offset,
                    'name': row['name'],
                })
        else:
            for lbl in sorted(np.unique(np.concatenate([y_tr, y_te]))):
                all_label_names.append({
                    'dataset': name,
                    'local_label': int(lbl),
                    'global_label': int(lbl) + offset,
                    'name': f'{name}_class_{lbl}',
                })

        n_local = len(np.unique(np.concatenate([y_tr, y_te])))
        label_offsets[name] = (offset, offset + n_local)
        y_tr = y_tr + offset
        y_te = y_te + offset
        offset += n_local

        x_train_parts.append(x_tr)
        y_train_parts.append(y_tr)
        x_test_parts.append(x_te)
        y_test_parts.append(y_te)
        print(f"  {name}: train={x_tr.shape}, test={x_te.shape}, "
              f"labels {label_offsets[name][0]}~{label_offsets[name][1]-1}")

    if not x_train_parts:
        print("没有可合并的数据集。")
        return

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    # 统一为 float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # 打乱训练集
    rng = np.random.RandomState(42)
    train_idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    np.save(os.path.join(OUT_DIR, 'x_train.npy'), x_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

    n_classes = offset
    print(f"\n合并完成: {n_classes} 类  |  "
          f"x_train: {x_train.shape}  |  x_test: {x_test.shape}")

    # 保存全局标签映射
    mapping_df = pd.DataFrame(all_label_names)
    mapping_df.to_csv(os.path.join(OUT_DIR, 'merged_label_mapping.csv'), index=False)

    # 保存元信息 JSON（供 train.py/demo.py 自动读取）
    global_names = [''] * n_classes
    for item in all_label_names:
        global_names[item['global_label']] = item['name']
    meta = {
        'num_classes': n_classes,
        'class_names': global_names,
        'label_offsets': {k: list(v) for k, v in label_offsets.items()},
    }
    with open(os.path.join(OUT_DIR, 'dataset_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元信息已保存到 {OUT_DIR}dataset_meta.json")
    print(f"标签映射已保存到 {OUT_DIR}merged_label_mapping.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['gym_gesture', 'recofit', 'mmfit'],
                        help='要合并的数据集名称列表')
    args = parser.parse_args()
    print(f"合并数据集: {args.datasets}")
    print("=" * 50)
    merge_datasets(args.datasets)
