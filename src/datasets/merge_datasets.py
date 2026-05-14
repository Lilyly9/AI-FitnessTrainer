"""
多数据集合并脚本
================================================================
将多个已预处理的数据集合并为统一的训练/测试集。

每个数据集需先单独运行其预处理脚本，生成：
  data/processed/{dataset_name}_x_train.npy
  data/processed/{dataset_name}_y_train.npy
  data/processed/{dataset_name}_x_test.npy
  data/processed/{dataset_name}_y_test.npy

然后运行此脚本合并所有数据集。

使用方法:
  python src/datasets/merge_datasets.py --datasets gym_gesture recofit recgym

输出: data/processed/x_train.npy, y_train.npy, x_test.npy, y_test.npy
"""

import numpy as np
import os
import argparse

OUT_DIR = 'data/processed/'


def merge_datasets(dataset_names):
    x_train_parts, y_train_parts = [], []
    x_test_parts, y_test_parts = [], []
    label_offsets = {}
    offset = 0

    for name in dataset_names:
        train_x_path = os.path.join(OUT_DIR, f'{name}_x_train.npy')
        train_y_path = os.path.join(OUT_DIR, f'{name}_y_train.npy')
        test_x_path = os.path.join(OUT_DIR, f'{name}_x_test.npy')
        test_y_path = os.path.join(OUT_DIR, f'{name}_y_test.npy')

        if not os.path.exists(train_x_path):
            print(f"跳过 {name}: 未找到预处理文件，请先运行 preprocess_{name}.py")
            continue

        x_tr = np.load(train_x_path)
        y_tr = np.load(train_y_path)
        x_te = np.load(test_x_path)
        y_te = np.load(test_y_path)

        # 标签偏移：每个数据集保持自己的标签空间
        label_offsets[name] = (offset, offset + len(np.unique(y_tr)))
        y_tr = y_tr + offset
        y_te = y_te + offset
        offset += len(np.unique(np.concatenate([y_tr, y_te])))

        x_train_parts.append(x_tr)
        y_train_parts.append(y_tr)
        x_test_parts.append(x_te)
        y_test_parts.append(y_te)

        print(f"  {name}: train={x_tr.shape}, test={x_te.shape}")

    if not x_train_parts:
        print("没有可合并的数据集。")
        return

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    # 打乱训练集
    train_idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    np.save(os.path.join(OUT_DIR, 'x_train.npy'), x_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

    print(f"\n合并完成:")
    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  x_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"  标签偏移: {label_offsets}")

    # 保存标签映射
    with open(os.path.join(OUT_DIR, 'label_mapping.txt'), 'w') as f:
        for name, (lo, hi) in label_offsets.items():
            f.write(f"{name}: labels {lo}~{hi - 1}\n")
    print(f"  标签映射已保存到 data/processed/label_mapping.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['gym_gesture'],
                        help='要合并的数据集名称列表')
    args = parser.parse_args()
    print(f"合并数据集: {args.datasets}")
    print("=" * 50)
    merge_datasets(args.datasets)
