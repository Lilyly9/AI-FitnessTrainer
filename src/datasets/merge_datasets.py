"""
多数据集合并脚本 — 所有数据集文件统一放在 data/processed/ 下。

文件命名规则: {dataset_name}_{x_train,y_train,x_test,y_test}.npy
标签映射:     {dataset_name}_label_mapping.csv

用法:
  python src/datasets/merge_datasets.py --datasets gym_gesture recofit mmfit
"""

import numpy as np
import pandas as pd
import os
import argparse
import json
import re

OUT_DIR = 'data/processed/'

# ═══════════════════════════════════════════════════════════════
# 跨数据集名称标准化：同一个动作在不同数据集中可能有不同的拼写、
# 大小写、单复数形式，通过此映射统一到同一个 canonical name。
# 不在映射中的名称保持原样，不会错误合并。
# ═══════════════════════════════════════════════════════════════

# 每个 canonical_name 对应的别名集合（均为 _normalize 后的形式）
CROSS_DATASET_ALIASES = {
    # squat 系列
    'squat':          {'squats', 'squat'},
    # pushup 系列 — pushup (variation) 是不同动作，不在此列
    'pushup':         {'pushups', 'pushup'},
    # bicep curl 系列 — Two-arm Dumbbell Curl / Alternating Dumbbell Curl 是变体，保持独立
    'bicep_curl':     {'bicep curls', 'bicep curl', 'bicep curls'},
    # jumping jack
    'jumping_jack':   {'jumping jacks', 'jumping jack', 'jumping jacks'},
    # situp 系列 — Sit-up (hands behind) / Butterfly Sit-up 是变体，保持独立
    'situp':          {'situps', 'sit ups', 'sit up', 'situp'},
    # lateral raise
    'lateral_raise':  {'lateral shoulder raises', 'lateral raise', 'lateral shoulder raise'},
    # tricep extension
    'tricep_extension': {'tricep extension', 'tricep extensions'},
    # lunge — Walking lunge 是不同变体，保持独立
    'lunge':          {'lunges', 'lunge'},
    # dumbbell row — Dumbbell Row (right)/(left) 是单侧变体，保持独立
    'dumbbell_row':   {'dumbbell rows', 'dumbbell row', 'dumbbell rows'},
    # dumbbell shoulder press
    'dumbbell_shoulder_press': {'dumbbell shoulder press', 'dumbbell shoulder presses'},
}


def _normalize_name(name):
    """将动作名称标准化为小写、去下划线/连字符、合并多余空格的规范形式。"""
    name = name.lower().strip()
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _to_canonical(normalized_name):
    """将标准化后的名称映射到 canonical name，无匹配则返回原值。"""
    for canonical, aliases in CROSS_DATASET_ALIASES.items():
        if normalized_name in aliases:
            return canonical
    return normalized_name


def find_dataset_files(name):
    """查找数据集文件（前缀格式）。"""
    tx = os.path.join(OUT_DIR, f'{name}_x_train.npy')
    ty = os.path.join(OUT_DIR, f'{name}_y_train.npy')
    ex = os.path.join(OUT_DIR, f'{name}_x_test.npy')
    ey = os.path.join(OUT_DIR, f'{name}_y_test.npy')
    if os.path.exists(tx):
        return tx, ty, ex, ey
    return None, None, None, None


def merge_datasets(dataset_names):
    # ── Pass 1: 按标准化名称去重，构建 canonical_name → global_label 共享映射 ──
    canonical_to_global = {}
    next_global_label = 0

    for name in dataset_names:
        tx, ty, ex, ey = find_dataset_files(name)
        if tx is None:
            continue
        mapping_path = os.path.join(OUT_DIR, f'{name}_label_mapping.csv')
        if os.path.exists(mapping_path):
            mapping = pd.read_csv(mapping_path)
            for _, row in mapping.iterrows():
                label_name = row['name']
                normalized = _normalize_name(label_name)
                canonical = _to_canonical(normalized)
                if canonical not in canonical_to_global:
                    canonical_to_global[canonical] = next_global_label
                    next_global_label += 1
                    print(f"  新增类别 [{canonical_to_global[canonical]}] {canonical}"
                          f"{' ← ' + normalized if normalized != canonical else ''}")
        else:
            # 无映射文件时，用 dataset_class_N 占位（天然唯一，不会碰撞）
            y_tr = np.load(ty)
            y_te = np.load(ey)
            for lbl in sorted(np.unique(np.concatenate([y_tr, y_te]))):
                label_name = f'{name}_class_{lbl}'
                if label_name not in canonical_to_global:
                    canonical_to_global[label_name] = next_global_label
                    next_global_label += 1

    # ── Pass 2: 加载数据，通过 canonical name 将 local label 映射到 shared global label ──
    x_train_parts, y_train_parts = [], []
    x_test_parts, y_test_parts = [], []
    label_sets = {}          # dataset → set of global labels
    all_label_names = []

    for ds_name in dataset_names:
        tx, ty, ex, ey = find_dataset_files(ds_name)
        if tx is None:
            print(f"跳过 {ds_name}: 未找到预处理文件")
            continue

        x_tr = np.load(tx)
        y_tr = np.load(ty)
        x_te = np.load(ex)
        y_te = np.load(ey)

        mapping_path = os.path.join(OUT_DIR, f'{ds_name}_label_mapping.csv')
        local_to_global = {}

        if os.path.exists(mapping_path):
            mapping = pd.read_csv(mapping_path)
            for _, row in mapping.iterrows():
                local_lbl = int(row['label'])
                label_name = row['name']
                canonical = _to_canonical(_normalize_name(label_name))
                global_lbl = canonical_to_global[canonical]
                local_to_global[local_lbl] = global_lbl
                all_label_names.append({
                    'dataset': ds_name,
                    'local_label': local_lbl,
                    'global_label': global_lbl,
                    'name': label_name,
                    'canonical': canonical,
                })
        else:
            for lbl in sorted(np.unique(np.concatenate([y_tr, y_te]))):
                label_name = f'{ds_name}_class_{lbl}'
                global_lbl = canonical_to_global[label_name]
                local_to_global[lbl] = global_lbl
                all_label_names.append({
                    'dataset': ds_name,
                    'local_label': int(lbl),
                    'global_label': global_lbl,
                    'name': label_name,
                    'canonical': label_name,
                })

        # 向量化重映射
        remap = np.vectorize(local_to_global.get)
        y_tr = remap(y_tr)
        y_te = remap(y_te)

        label_sets[ds_name] = sorted(local_to_global.values())

        x_train_parts.append(x_tr)
        y_train_parts.append(y_tr)
        x_test_parts.append(x_te)
        y_test_parts.append(y_te)
        print(f"  {ds_name}: train={x_tr.shape}, test={x_te.shape}, "
              f"labels {label_sets[ds_name]}")

    if not x_train_parts:
        print("没有可合并的数据集。")
        return

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    # 统一 dtype
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

    n_classes = len(canonical_to_global)
    print(f"\n合并完成: {n_classes} 类  |  "
          f"x_train: {x_train.shape}  |  x_test: {x_test.shape}")

    # 保存全局标签映射
    mapping_df = pd.DataFrame(all_label_names)
    mapping_df.to_csv(os.path.join(OUT_DIR, 'merged_label_mapping.csv'), index=False)

    # 保存元信息 JSON — class_names 使用 canonical 名称确保唯一
    global_names = [''] * n_classes
    seen_globals = set()
    for item in all_label_names:
        gid = item['global_label']
        if gid not in seen_globals:
            global_names[gid] = item.get('canonical', item['name'])
            seen_globals.add(gid)
    meta = {
        'num_classes': n_classes,
        'class_names': global_names,
        'label_sets': {k: list(v) for k, v in label_sets.items()},
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
