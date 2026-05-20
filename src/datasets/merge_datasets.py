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
from collections import Counter

OUT_DIR = 'data/processed/'

# ============================================================
# 跨数据集名称标准化
# ============================================================
CROSS_DATASET_ALIASES = {
    'squat':          {'squats', 'squat'},
    'pushup':         {'pushups', 'pushup'},
    'bicep_curl':     {'bicep curls', 'bicep curl', 'bicep curls'},
    'jumping_jack':   {'jumping jacks', 'jumping jack', 'jumping jacks'},
    'situp':          {'situps', 'sit ups', 'sit up', 'situp'},
    'lateral_raise':  {'lateral shoulder raises', 'lateral raise', 'lateral shoulder raise'},
    'tricep_extension': {'tricep extension', 'tricep extensions'},
    'lunge':          {'lunges', 'lunge'},
    'dumbbell_row':   {'dumbbell rows', 'dumbbell row', 'dumbbell rows'},
    'dumbbell_shoulder_press': {'dumbbell shoulder press', 'dumbbell shoulder presses'},
}


def _normalize_name(name):
    name = name.lower().strip()
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _to_canonical(normalized_name):
    for canonical, aliases in CROSS_DATASET_ALIASES.items():
        if normalized_name in aliases:
            return canonical
    return normalized_name


def find_dataset_files(name):
    tx = os.path.join(OUT_DIR, f'{name}_x_train.npy')
    ty = os.path.join(OUT_DIR, f'{name}_y_train.npy')
    ex = os.path.join(OUT_DIR, f'{name}_x_test.npy')
    ey = os.path.join(OUT_DIR, f'{name}_y_test.npy')
    dt = os.path.join(OUT_DIR, f'{name}_domain_train.npy')
    de = os.path.join(OUT_DIR, f'{name}_domain_test.npy')
    if os.path.exists(tx):
        return tx, ty, ex, ey, dt, de
    return None, None, None, None, None, None


def merge_datasets(dataset_names):
    # --- Pass 1: build canonical_name -> global_label map ---
    canonical_to_global = {}
    next_global_label = 0

    for name in dataset_names:
        tx, ty, ex, ey, dt, de = find_dataset_files(name)
        if tx is None:
            continue
        mapping_path = os.path.join(OUT_DIR, f'{name}_label_mapping.csv')
        if os.path.exists(mapping_path):
            mapping = pd.read_csv(mapping_path)
            for _, row in mapping.iterrows():
                # 兼容不同命名的列：name / activity_name
                label_name = row.get('name', row.get('activity_name', f'{name}_class_{row.iloc[0]}'))
                normalized = _normalize_name(label_name)
                canonical = _to_canonical(normalized)
                if canonical not in canonical_to_global:
                    canonical_to_global[canonical] = next_global_label
                    next_global_label += 1
                    print(f"  new class [{canonical_to_global[canonical]}] {canonical}"
                          f"{' <- ' + normalized if normalized != canonical else ''}")
        else:
            y_tr = np.load(ty); y_te = np.load(ey)
            for lbl in sorted(np.unique(np.concatenate([y_tr, y_te]))):
                label_name = f'{name}_class_{lbl}'
                if label_name not in canonical_to_global:
                    canonical_to_global[label_name] = next_global_label
                    next_global_label += 1

    # --- Pass 2: load data, remap local labels to global ---
    x_train_parts, y_train_parts = [], []
    x_test_parts, y_test_parts = [], []
    label_sets = {}
    all_label_names = []

    # Domain label parts
    domain_train_parts, domain_test_parts = [], []
    dataset_domain_map = {}  # ds_name -> domain_id

    for ds_name in dataset_names:
        tx, ty, ex, ey, dt, de = find_dataset_files(ds_name)
        if tx is None:
            print(f"skip {ds_name}: no preprocessed files found")
            continue

        x_tr = np.load(tx); y_tr = np.load(ty)
        x_te = np.load(ex); y_te = np.load(ey)

        # Load domain labels (default to -1 if missing)
        d_tr = np.load(dt) if os.path.exists(dt) else np.full(len(y_tr), -1, dtype=np.int64)
        d_te = np.load(de) if os.path.exists(de) else np.full(len(y_te), -1, dtype=np.int64)
        domain_id = int(d_tr[0]) if len(d_tr) > 0 else -1
        dataset_domain_map[ds_name] = domain_id
        print(f"  {ds_name} domain_id={domain_id} ({'forearm' if domain_id==0 else 'wrist' if domain_id==1 else 'unknown'})")

        mapping_path = os.path.join(OUT_DIR, f'{ds_name}_label_mapping.csv')
        local_to_global = {}

        if os.path.exists(mapping_path):
            mapping = pd.read_csv(mapping_path)
            for _, row in mapping.iterrows():
                local_lbl = int(row.get('label', row.get('label_id', 0)))
                label_name = row.get('name', row.get('activity_name', f'class_{local_lbl}'))
                canonical = _to_canonical(_normalize_name(label_name))
                global_lbl = canonical_to_global[canonical]
                local_to_global[local_lbl] = global_lbl
                all_label_names.append({
                    'dataset': ds_name, 'local_label': local_lbl,
                    'global_label': global_lbl, 'name': label_name,
                    'canonical': canonical,
                })
        else:
            for lbl in sorted(np.unique(np.concatenate([y_tr, y_te]))):
                label_name = f'{ds_name}_class_{lbl}'
                global_lbl = canonical_to_global[label_name]
                local_to_global[lbl] = global_lbl
                all_label_names.append({
                    'dataset': ds_name, 'local_label': int(lbl),
                    'global_label': global_lbl, 'name': label_name,
                    'canonical': label_name,
                })

        remap = np.vectorize(local_to_global.get)
        y_tr = remap(y_tr); y_te = remap(y_te)
        label_sets[ds_name] = sorted(local_to_global.values())
        x_train_parts.append(x_tr); y_train_parts.append(y_tr)
        x_test_parts.append(x_te); y_test_parts.append(y_te)
        domain_train_parts.append(d_tr); domain_test_parts.append(d_te)
        print(f"  {ds_name}: train={x_tr.shape}, test={x_te.shape}, labels {label_sets[ds_name]}")

    if not x_train_parts:
        print("No datasets to merge.")
        return

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    d_train = np.concatenate(domain_train_parts, axis=0)
    d_test = np.concatenate(domain_test_parts, axis=0)

    x_train = x_train.astype(np.float32); x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int64); y_test = y_test.astype(np.int64)
    d_train = d_train.astype(np.int64); d_test = d_test.astype(np.int64)

    n_classes = len(canonical_to_global)

    # NOTE: 保留各数据集独立的 Min-Max 归一化结果，不做全局重归一化。
    # 不同数据集使用不同传感器，传感器量程和零点不同，独立归一化
    # 将这些差异转化为对模型有用的"数据集特征"，而非强行抹除。

    # ============================================================
    # Step B: Stratified split (guarantees every class in both train & test)
    # ============================================================
    train_cnt = Counter(y_train)
    test_cnt = Counter(y_test)
    zero_test = [c for c in range(n_classes) if test_cnt.get(c, 0) == 0]
    zero_train = [c for c in range(n_classes) if train_cnt.get(c, 0) == 0]

    if zero_test or zero_train:
        print(f"\nWARNING: split issue detected -> auto-fix stratified split")
        print(f"  classes missing from test: {zero_test}")
        print(f"  classes missing from train: {zero_train}")

        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        d_all = np.concatenate([d_train, d_test], axis=0)

        from sklearn.model_selection import train_test_split
        try:
            idx_tr, idx_te = train_test_split(
                np.arange(len(y_all)), test_size=0.2, random_state=42, stratify=y_all)
            x_train, x_test = x_all[idx_tr], x_all[idx_te]
            y_train, y_test = y_all[idx_tr], y_all[idx_te]
            d_train, d_test = d_all[idx_tr], d_all[idx_te]
        except ValueError:
            # Some classes have only 1 sample, stratify fails, do manual fix
            print("  stratify failed (some classes too small), manual fix...")
            test_size = max(int(len(y_all) * 0.2), n_classes)
            x_train, x_test, y_train, y_test = train_test_split(
                x_all, y_all, test_size=test_size, random_state=42)
            d_train = np.ones(len(x_train), dtype=np.int64) * -1

        print(f"  after re-split: train={x_train.shape}, test={x_test.shape}")

        new_test_cnt = Counter(y_test)
        still_zero = [c for c in range(n_classes) if new_test_cnt.get(c, 0) == 0]
        if still_zero:
            print(f"  still {len(still_zero)} classes missing, forced allocation...")
            for cls in still_zero:
                cls_idx = np.where(y_all == cls)[0]
                if len(cls_idx) >= 2:
                    x_train = np.delete(x_all, cls_idx[-1], axis=0)
                    y_train = np.delete(y_all, cls_idx[-1])
                    x_test = x_all[cls_idx[-1:]]
                    y_test = y_all[cls_idx[-1:]]
                    print(f"    forced class {cls} into test set")
    else:
        print(f"\nOK: all {n_classes} classes present in both train & test")

    # ============================================================
    # Step C: Class distribution report
    # ============================================================
    train_cnt = Counter(y_train)
    test_cnt = Counter(y_test)
    print(f"\n=== Class Distribution Report ===")
    print(f"  Train: min={min(train_cnt.values())} (cls {min(train_cnt, key=train_cnt.get)}), "
          f"max={max(train_cnt.values())} (cls {max(train_cnt, key=train_cnt.get)}), "
          f"ratio={max(train_cnt.values())/max(min(train_cnt.values()),1):.1f}x")
    print(f"  Test:  min={min(test_cnt.values())}, max={max(test_cnt.values())}")
    print(f"  Classes with <10 test samples: "
          f"{sum(1 for c in range(n_classes) if test_cnt.get(c,0) < 10)}")
    print(f"  Classes with <50 train samples: "
          f"{sum(1 for c in range(n_classes) if train_cnt.get(c,0) < 50)}")

    # ============================================================
    # Shuffle & Save
    # ============================================================
    rng = np.random.RandomState(42)
    train_idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    np.save(os.path.join(OUT_DIR, 'x_train.npy'), x_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(OUT_DIR, 'domain_train.npy'), d_train)
    np.save(os.path.join(OUT_DIR, 'domain_test.npy'), d_test)

    print(f"\nMerge complete: {n_classes} classes | "
          f"x_train: {x_train.shape} | x_test: {x_test.shape}")

    # Save label mapping
    mapping_df = pd.DataFrame(all_label_names)
    mapping_df.to_csv(os.path.join(OUT_DIR, 'merged_label_mapping.csv'), index=False)

    # Save meta JSON
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
        'domain_info': {
            'num_domains': 2,
            'domain_names': ['forearm', 'wrist'],
            'dataset_domains': dataset_domain_map,
        },
    }
    with open(os.path.join(OUT_DIR, 'dataset_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Meta saved to {OUT_DIR}dataset_meta.json")
    print(f"Label mapping saved to {OUT_DIR}merged_label_mapping.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['gym_gesture', 'recofit', 'mmfit'])
    args = parser.parse_args()
    print(f"Merging: {args.datasets}")
    print("=" * 50)
    merge_datasets(args.datasets)
