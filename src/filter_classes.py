"""
自动类别过滤脚本 — 基于模型评估结果智能筛除难以学习的类别。

过滤策略（按优先级）:
  1. train < min_train (默认50): 样本太少，直接移除
  2. test < min_test (默认5):   测试样本太少，无法评估
  3. F1 = 0 AND train < safe_train (默认500): 学不到 + 数据不够
  4. F1 < min_f1 (默认0.05):   完全无法识别

保存 clean 版本的数据集、映射和元信息。

用法:
  python -X utf8 src/filter_classes.py                                    # 默认自动过滤
  python -X utf8 src/filter_classes.py --min-train 100 --min-f1 0.05      # 自定义阈值
  python -X utf8 src/filter_classes.py --dry-run                          # 仅分析不执行
  python -X utf8 src/filter_classes.py --keep gym_gesture                 # 保留某数据集全部类
"""

import numpy as np
import json
import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = 'data/processed/'
OUT_DIR = DATA_DIR


def load_model_eval(checkpoint_path='models/gesture_50.pth'):
    """加载已训练的模型并生成逐类评估。"""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report

    if not os.path.exists(checkpoint_path):
        return None

    from model_v2 import Gesture1DCNN
    from dataset import GestureDataset

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt)

    # Detect model type
    if 'conv1.weight' in sd:
        n_cls = sd['fc.weight'].shape[0]
        model = Gesture1DCNN(num_classes=n_cls)
        model.load_state_dict(sd)
    else:
        print(f"Warning: checkpoint {checkpoint_path} not Gesture1DCNN, skip eval")
        return None

    model.eval()

    test_ds = GestureDataset(DATA_DIR + 'x_test.npy', DATA_DIR + 'y_test.npy', train=False)
    test_loader = DataLoader(test_ds, batch_size=64)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_preds.extend(model(x).argmax(1).tolist())
            all_labels.extend(y.tolist())

    with open(DATA_DIR + 'dataset_meta.json') as f:
        meta = json.load(f)

    rpt = classification_report(
        all_labels, all_preds,
        labels=list(range(meta['num_classes'])),
        target_names=meta['class_names'],
        output_dict=True, zero_division=0)
    return {name: rpt[name]['f1-score'] for name in meta['class_names'] if name in rpt}


def analyze_classes(meta, train_counts, test_counts, f1_scores=None):
    """分析每个类的可学习性，返回建议过滤列表。"""
    class_names = meta['class_names']
    n_cls = meta['num_classes']

    # 确定每个类来自哪个数据集
    cls_source = {}
    for ds, lbls in meta['label_sets'].items():
        for l in lbls:
            cls_source.setdefault(l, []).append(ds)

    analysis = []
    for cls_id in range(n_cls):
        name = class_names[cls_id]
        trc = train_counts.get(cls_id, 0)
        tec = test_counts.get(cls_id, 0)
        f1 = f1_scores.get(name, -1) if f1_scores else -1
        src = ','.join(cls_source.get(cls_id, ['?']))
        analysis.append({
            'id': cls_id, 'name': name, 'train': trc, 'test': tec,
            'f1': f1, 'source': src,
        })

    return analysis


def apply_filters(analysis, args, meta):
    """应用过滤规则，返回 (保留列表, 移除列表, 移除原因)。"""
    removed = []
    reasons = {}

    for info in analysis:
        cls_id = info['id']
        name = info['name']
        trc = info['train']
        tec = info['test']
        f1 = info['f1']
        src = info['source']

        rlist = []

        # Rule 1: train 样本太少
        if trc < args.min_train:
            rlist.append(f'train={trc}<{args.min_train}')

        # Rule 2: test 样本太少
        if tec < args.min_test:
            rlist.append(f'test={tec}<{args.min_test}')

        # Rule 3: F1=0 但 train 不够多（给有数据的类第二次机会）
        if f1 == 0 and trc < args.safe_train:
            rlist.append(f'F1=0&train={trc}<{args.safe_train}')

        # Rule 4: F1 太低
        if f1 >= 0 and f1 < args.min_f1:
            rlist.append(f'F1={f1:.3f}<{args.min_f1}')

        # Rule 5: 极低样本量且来自单一数据集 + F1=0
        if trc < 100 and f1 == 0 and len(info['source'].split(',')) == 1:
            # 不影响已标记的（已被其他规则命中）
            pass  # 不额外增加规则，靠 Rule 1/3 覆盖

        if rlist:
            removed.append(cls_id)
            reasons[cls_id] = ' | '.join(rlist)

    keep = sorted(set(range(meta['num_classes'])) - set(removed))

    # 确保 --keep 指定的数据集全部保留
    if args.keep_datasets:
        for ds in args.keep_datasets:
            if ds in meta['label_sets']:
                for l in meta['label_sets'][ds]:
                    if l in removed:
                        removed.remove(l)
                        if l in reasons:
                            del reasons[l]
        keep = sorted(set(range(meta['num_classes'])) - set(removed))

    return keep, removed, reasons


def main():
    parser = argparse.ArgumentParser(description='Auto-filter hard classes from merged dataset')

    # Filter thresholds
    parser.add_argument('--min-train', type=int, default=50,
                        help='Min training samples per class (default: 50)')
    parser.add_argument('--min-test', type=int, default=5,
                        help='Min test samples per class (default: 5)')
    parser.add_argument('--safe-train', type=int, default=500,
                        help='Min training samples to keep a class despite F1=0 (default: 500)')
    parser.add_argument('--min-f1', type=float, default=0.05,
                        help='Min F1 score to keep a class (default: 0.05)')

    # Keep options
    parser.add_argument('--keep-datasets', nargs='+', default=[],
                        help='Datasets to keep entirely (e.g., gym_gesture)')
    parser.add_argument('--keep-classes', nargs='+', default=[],
                        help='Specific class names to keep regardless')

    # Checkpoint
    parser.add_argument('--checkpoint', default='models/gesture_50.pth',
                        help='Model checkpoint for F1 evaluation')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip model evaluation, use sample counts only')

    # Output
    parser.add_argument('--dry-run', action='store_true',
                        help='Only analyze, do not write files')

    args = parser.parse_args()

    # ---- Load data ----
    with open(DATA_DIR + 'dataset_meta.json') as f:
        meta = json.load(f)

    y_train = np.load(DATA_DIR + 'y_train.npy').flatten()
    y_test = np.load(DATA_DIR + 'y_test.npy').flatten()
    x_train = np.load(DATA_DIR + 'x_train.npy')
    x_test = np.load(DATA_DIR + 'x_test.npy')

    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    # ---- Model evaluation ----
    f1_scores = None
    if not args.no_eval and os.path.exists(args.checkpoint):
        f1_scores = load_model_eval(args.checkpoint)
        if f1_scores:
            print(f"Model evaluation loaded: {len(f1_scores)} classes")
    else:
        print(f"No model evaluation (--no-eval or checkpoint not found)")

    # ---- Analyze ----
    analysis = analyze_classes(meta, train_counts, test_counts, f1_scores)

    # Override keep list
    keep_classes_manual = set(args.keep_classes)

    # ---- Apply filters ----
    keep, removed, reasons = apply_filters(analysis, args, meta)

    # Manual keep
    for name in keep_classes_manual:
        for info in analysis:
            if info['name'] == name and info['id'] in removed:
                removed.remove(info['id'])
                del reasons[info['id']]
    keep = sorted(set(range(meta['num_classes'])) - set(removed))

    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"FILTER ANALYSIS")
    print(f"{'='*60}")
    print(f"Original classes: {meta['num_classes']}")
    print(f"Classes to KEEP:  {len(keep)}")
    print(f"Classes to REMOVE: {len(removed)}")
    print(f"Training data kept: {sum(train_counts.get(c,0) for c in keep)}/{len(y_train)} "
          f"({sum(train_counts.get(c,0) for c in keep)/len(y_train)*100:.1f}%)")
    print(f"Test data kept:     {sum(test_counts.get(c,0) for c in keep)}/{len(y_test)} "
          f"({sum(test_counts.get(c,0) for c in keep)/len(y_test)*100:.1f}%)")

    print(f"\n--- Removed classes ---")
    for cls_id in removed:
        info = [a for a in analysis if a['id'] == cls_id][0]
        src = info['source']
        f1_str = f"f1={info['f1']:.3f}" if info['f1'] >= 0 else 'no_eval'
        reason = reasons.get(cls_id, 'manual')
        print(f"  [{cls_id:2d}] {info['name']:<45s} "
              f"train={info['train']:4d} test={info['test']:3d} "
              f"{f1_str:<12s} src={src}  |  {reason}")

    print(f"\n--- Kept classes ---")
    for cls_id in keep:
        info = [a for a in analysis if a['id'] == cls_id][0]
        src = info['source']
        f1_val = info['f1']
        f1_str = f'{f1_val:.3f}' if f1_val >= 0 else '?'
        print(f"  [{cls_id:2d}] {info['name']:<45s} "
              f"train={info['train']:4d} test={info['test']:3d} "
              f"f1={f1_str:<8s} src={src}")

    # Per-dataset summary
    print(f"\n--- Per-dataset summary ---")
    for ds, lbls in meta['label_sets'].items():
        kept = [l for l in lbls if l in keep]
        removed_ds = [l for l in lbls if l in removed]
        print(f"  {ds}: kept {len(kept)}/{len(lbls)}, removed {len(removed_ds)}/{len(lbls)}")

    if args.dry_run:
        print(f"\n[DRY RUN] No files written. Remove --dry-run to apply.")
        return

    # ---- Apply: rewrite merged files ----
    print(f"\n{'='*60}")
    print(f"APPLYING FILTERS...")
    print(f"{'='*60}")

    # Build new label mapping (compact: 0..N-1)
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_class_names = [meta['class_names'][old] for old in keep]

    # Filter and remap
    tr_mask = np.isin(y_train, keep)
    te_mask = np.isin(y_test, keep)

    x_train_new = x_train[tr_mask]
    y_train_old = y_train[tr_mask]
    x_test_new = x_test[te_mask]
    y_test_old = y_test[te_mask]

    remap = np.vectorize(old_to_new.get)
    y_train_new = remap(y_train_old).astype(np.int64)
    y_test_new = remap(y_test_old).astype(np.int64)

    # Shuffle train
    rng = np.random.RandomState(42)
    tr_idx = rng.permutation(len(x_train_new))
    x_train_new, y_train_new = x_train_new[tr_idx], y_train_new[tr_idx]

    # Save
    np.save(OUT_DIR + 'x_train.npy', x_train_new.astype(np.float32))
    np.save(OUT_DIR + 'y_train.npy', y_train_new)
    np.save(OUT_DIR + 'x_test.npy', x_test_new.astype(np.float32))
    np.save(OUT_DIR + 'y_test.npy', y_test_new)

    # Rebuild label_sets
    new_label_sets = {}
    for ds, lbls in meta['label_sets'].items():
        new_ds_lbls = sorted([old_to_new[l] for l in lbls if l in keep])
        if new_ds_lbls:
            new_label_sets[ds] = new_ds_lbls

    # Save meta
    new_meta = {
        'num_classes': len(keep),
        'class_names': new_class_names,
        'label_sets': new_label_sets,
        'removed_classes': [meta['class_names'][r] for r in removed],
        'removed_reasons': {meta['class_names'][r]: reasons.get(r, 'manual') for r in removed},
    }
    with open(OUT_DIR + 'dataset_meta.json', 'w', encoding='utf-8') as f:
        json.dump(new_meta, f, indent=2, ensure_ascii=False)

    # Rebuild merged mapping
    import pandas as pd
    old_mapping = pd.read_csv(OUT_DIR + 'merged_label_mapping.csv') if \
        os.path.exists(OUT_DIR + 'merged_label_mapping.csv') else None
    if old_mapping is not None:
        old_mapping['keep'] = old_mapping['global_label'].isin(keep)
        old_mapping_kept = old_mapping[old_mapping['keep']].copy()
        old_mapping_kept['global_label'] = old_mapping_kept['global_label'].map(old_to_new)
        old_mapping_kept.drop(columns=['keep'], inplace=True)
        old_mapping_kept.to_csv(OUT_DIR + 'merged_label_mapping.csv', index=False)

    # Stats
    new_tr_cnt = Counter(y_train_new)
    new_te_cnt = Counter(y_test_new)
    print(f"\nFiltered dataset saved:")
    print(f"  x_train: {x_train_new.shape}  y_train: {y_train_new.shape}")
    print(f"  x_test:  {x_test_new.shape}   y_test:  {y_test_new.shape}")
    print(f"  Classes: {len(keep)}")
    print(f"  Train: min={min(new_tr_cnt.values())}, max={max(new_tr_cnt.values())}")
    print(f"  Test:  min={min(new_te_cnt.values())}, max={max(new_te_cnt.values())}")
    print(f"  Removed classes saved in dataset_meta.json.removed_classes")
    print(f"\nRe-train with: python -u -X utf8 src/train_v2.py --model Gesture1DCNN --epochs 100 --save_path models/filtered.pth")


if __name__ == '__main__':
    main()
