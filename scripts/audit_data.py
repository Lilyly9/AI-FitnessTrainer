"""
数据全面审计脚本 — 检查预处理后的 .npy 文件质量。
输出完整审计报告。
"""
import numpy as np
import json
import os
from collections import Counter

DATA_DIR = 'data/processed/'

def load(name):
    path = os.path.join(DATA_DIR, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None

print("=" * 60)
print("DATA AUDIT REPORT")
print("=" * 60)

# ---- 1. Basic shapes ----
x_train = load('x_train.npy')
y_train = load('y_train.npy')
x_test = load('x_test.npy')
y_test = load('y_test.npy')

if x_train is None:
    print("ERROR: x_train.npy not found!")
    exit(1)

y_train = y_train.flatten().astype(np.int64)
y_test = y_test.flatten().astype(np.int64)

print(f"\n[1] SHAPES")
print(f"  x_train: {x_train.shape}  dtype={x_train.dtype}  min={x_train.min():.4f}  max={x_train.max():.4f}  mean={x_train.mean():.4f}  std={x_train.std():.4f}")
print(f"  y_train: {y_train.shape}  dtype={y_train.dtype}")
print(f"  x_test:  {x_test.shape}   dtype={x_test.dtype}   min={x_test.min():.4f}  max={x_test.max():.4f}")
print(f"  y_test:  {y_test.shape}   dtype={y_test.dtype}")

# ---- 2. Label analysis ----
train_cnt = Counter(y_train)
test_cnt = Counter(y_test)
all_classes = sorted(set(list(train_cnt.keys()) + list(test_cnt.keys())))
n_cls = max(all_classes) + 1
print(f"\n[2] LABELS")
print(f"  Unique classes: {len(all_classes)}  (0-{max(all_classes)})")
print(f"  Train labels: {len(train_cnt)}  Test labels: {len(test_cnt)}")

zero_test = [c for c in range(n_cls) if test_cnt.get(c, 0) == 0]
zero_train = [c for c in range(n_cls) if train_cnt.get(c, 0) == 0]
print(f"  Missing from test: {len(zero_test)} classes {zero_test if len(zero_test) < 30 else ''}")
print(f"  Missing from train: {len(zero_train)} classes")

# Distribution stats
if train_cnt:
    tr_vals = list(train_cnt.values())
    print(f"  Train: min={min(tr_vals)}  max={max(tr_vals)}  median={np.median(tr_vals):.0f}  mean={np.mean(tr_vals):.0f}")
    print(f"  Imbalance ratio (max/min): {max(tr_vals)/max(min(tr_vals),1):.1f}x")
    rare = [(c, n) for c, n in sorted(train_cnt.items(), key=lambda x: x[1]) if n < 50]
    print(f"  Classes with <50 train samples: {len(rare)}")
    if rare:
        for c, n in rare[:15]:
            print(f"    class {c}: {n}")
        if len(rare) > 15:
            print(f"    ... and {len(rare)-15} more")

# ---- 3. NaN/Inf/Zero checks ----
print(f"\n[3] DATA QUALITY")
nan_tr = np.isnan(x_train).any(axis=(1,2)).sum()
nan_te = np.isnan(x_test).any(axis=(1,2)).sum()
inf_tr = np.isinf(x_train).any(axis=(1,2)).sum()
inf_te = np.isinf(x_test).any(axis=(1,2)).sum()
zero_win_tr = (np.abs(x_train).max(axis=(1,2)) < 1e-8).sum()
zero_win_te = (np.abs(x_test).max(axis=(1,2)) < 1e-8).sum()
print(f"  NaN windows:  train={nan_tr}  test={nan_te}")
print(f"  Inf windows:  train={inf_tr}  test={inf_te}")
print(f"  Zero windows: train={zero_win_tr}  test={zero_win_te}")

# Per-channel statistics
ch_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
print(f"\n  Per-channel stats (train):")
for c in range(min(6, x_train.shape[1])):
    ch = x_train[:, c, :]
    print(f"    {ch_names[c] if c < len(ch_names) else f'ch{c}'}: "
          f"mean={ch.mean():+.4f}  std={ch.std():.4f}  "
          f"min={ch.min():+.4f}  max={ch.max():+.4f}")

# ---- 4. Duplicate windows ----
print(f"\n[4] DUPLICATE WINDOWS")
# Flatten to 1D for efficient dedup
x_train_flat = x_train.reshape(x_train.shape[0], -1)
_, unique_idx = np.unique(x_train_flat, axis=0, return_index=True)
dup_count = len(x_train) - len(unique_idx)
print(f"  Train duplicate windows: {dup_count}/{len(x_train)} ({100*dup_count/max(len(x_train),1):.2f}%)")

x_test_flat = x_test.reshape(x_test.shape[0], -1)
_, unique_idx_te = np.unique(x_test_flat, axis=0, return_index=True)
dup_count_te = len(x_test) - len(unique_idx_te)
print(f"  Test duplicate windows: {dup_count_te}/{len(x_test)} ({100*dup_count_te/max(len(x_test),1):.2f}%)")

# Check overlapping windows (50% overlap due to STEP=100, WINDOW=200)
# This is expected and not a bug — just report
overlap_pct = 100 * (1 - 100/200)  # 50%
print(f"  NOTE: Window overlap (STRIDE=100, SIZE=200) = {overlap_pct:.0f}% — this is normal for sliding windows")

# ---- 5. Domain analysis ----
d_train = load('domain_train.npy')
d_test = load('domain_test.npy')
print(f"\n[5] DOMAIN LABELS")
if d_train is not None:
    d_train = d_train.flatten().astype(np.int64)
    d_test = d_test.flatten().astype(np.int64)
    d_tr_cnt = Counter(d_train)
    d_te_cnt = Counter(d_test)
    print(f"  Train domain distribution: {dict(sorted(d_tr_cnt.items()))}")
    print(f"  Test domain distribution:  {dict(sorted(d_te_cnt.items()))}")
    for d_id, d_name in {0: 'forearm', 1: 'wrist'}.items():
        tr_n = d_tr_cnt.get(d_id, 0)
        te_n = d_te_cnt.get(d_id, 0)
        print(f"  Domain {d_id} ({d_name}): train={tr_n}  test={te_n}  ratio tr/te={tr_n/max(te_n,1):.1f}")
else:
    print("  domain_train.npy NOT FOUND")

# ---- 6. Per-dataset breakdown using dataset_meta.json ----
print(f"\n[6] PER-DATASET BREAKDOWN")
meta_path = os.path.join(DATA_DIR, 'dataset_meta.json')
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    class_names = meta.get('class_names', [])
    label_sets = meta.get('label_sets', {})
    domain_info = meta.get('domain_info', {})

    for ds, lbls in label_sets.items():
        tr_mask = np.isin(y_train, lbls)
        te_mask = np.isin(y_test, lbls)
        tr_n = tr_mask.sum()
        te_n = te_mask.sum()
        print(f"  {ds}: train={tr_n}  test={te_n}  "
              f"train%={100*tr_n/max(len(y_train),1):.1f}%  "
              f"test%={100*te_n/max(len(y_test),1):.1f}%  "
              f"classes={len(lbls)}")

    print(f"\n  Domain info: {domain_info}")

    # Check feature range differences between wrist and forearm
    if d_train is not None:
        print(f"\n[7] DOMAIN GAP ANALYSIS (wrist vs forearm feature stats)")
        forearm_mask_tr = d_train == 0
        wrist_mask_tr = d_train == 1

        for domain_mask, domain_name in [(forearm_mask_tr, 'forearm'), (wrist_mask_tr, 'wrist')]:
            if domain_mask.sum() == 0:
                print(f"  {domain_name}: NO SAMPLES")
                continue
            xx = x_train[domain_mask]
            print(f"  {domain_name} (n={len(xx)}):")
            for c in range(min(6, xx.shape[1])):
                ch = xx[:, c, :]
                print(f"    {ch_names[c] if c < len(ch_names) else f'ch{c}'}: "
                      f"mean={ch.mean():+.4f}  std={ch.std():.4f}")

        # Per-channel mean difference between domains
        if forearm_mask_tr.sum() > 0 and wrist_mask_tr.sum() > 0:
            print(f"\n  Per-channel mean difference (wrist - forearm):")
            for c in range(min(6, x_train.shape[1])):
                diff = x_train[wrist_mask_tr, c, :].mean() - x_train[forearm_mask_tr, c, :].mean()
                sign = "LARGE" if abs(diff) > 0.3 else ("medium" if abs(diff) > 0.1 else "small")
                print(f"    {ch_names[c] if c < len(ch_names) else f'ch{c}'}: {diff:+.4f}  [{sign}]")
else:
    print("  dataset_meta.json NOT FOUND")

# ---- 8. Class distribution tail ----
print(f"\n[8] CLASS DISTRIBUTION TAIL (bottom 20 classes by train count)")
tail = sorted(train_cnt.items(), key=lambda x: x[1])[:20]
for c, n in tail:
    name = class_names[c] if c < len(class_names) else f'class_{c}'
    te_n = test_cnt.get(c, 0)
    print(f"  [{c:3d}] {name:<40s} train={n:5d}  test={te_n:5d}  {'⚠ <50' if n < 50 else ''}  {'⚠ test=0' if te_n == 0 else ''}")

# ---- 9. PCA check on sample ----
print(f"\n[9] PCA PREVIEW (first 2 components of 5000 random train samples)")
from sklearn.decomposition import PCA
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(x_train), min(5000, len(x_train)), replace=False)
sample_flat = x_train[sample_idx].reshape(len(sample_idx), -1)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(sample_flat)
print(f"  PCA variance ratio: {pca.explained_variance_ratio_}  (total={pca.explained_variance_ratio_.sum():.3f})")
print(f"  PCA component 1 range: [{pca_result[:,0].min():.2f}, {pca_result[:,0].max():.2f}]")
print(f"  PCA component 2 range: [{pca_result[:,1].min():.2f}, {pca_result[:,1].max():.2f}]")
if d_train is not None and len(d_train) == len(x_train):
    forearm_in_sample = d_train[sample_idx] == 0
    wrist_in_sample = d_train[sample_idx] == 1
    if forearm_in_sample.sum() > 0 and wrist_in_sample.sum() > 0:
        f_center = pca_result[forearm_in_sample].mean(axis=0)
        w_center = pca_result[wrist_in_sample].mean(axis=0)
        gap = np.linalg.norm(f_center - w_center)
        print(f"  PCA domain gap (L2 distance between centers): {gap:.3f}")

print(f"\n[10] RECOFIT-SPECIFIC CHECK")
# Check if RecoFit data has different shape (post-PCA it should still be 6x200)
recofit_x_tr = load('recofit_x_train.npy')
if recofit_x_tr is not None:
    print(f"  recofit_x_train.npy: {recofit_x_tr.shape}  dtype={recofit_x_tr.dtype}")
    print(f"  min={recofit_x_tr.min():.4f}  max={recofit_x_tr.max():.4f}  mean={recofit_x_tr.mean():.4f}")
    # Check if PCA channels look different from raw
    print(f"  ch0 range: [{recofit_x_tr[:,0,:].min():.4f}, {recofit_x_tr[:,0,:].max():.4f}]")
    print(f"  ch1 range: [{recofit_x_tr[:,1,:].min():.4f}, {recofit_x_tr[:,1,:].max():.4f}]")
else:
    print("  recofit_x_train.npy NOT FOUND (may have been merged only)")

# Check individual datasets
for ds in ['gym_gesture', 'mmfit', 'recofit']:
    ds_x_tr = load(f'{ds}_x_train.npy')
    if ds_x_tr is not None:
        ds_y_tr = load(f'{ds}_y_train.npy')
        ds_x_te = load(f'{ds}_x_test.npy')
        ds_y_te = load(f'{ds}_y_test.npy')
        print(f"\n  {ds}:")
        if ds_y_tr is not None:
            print(f"    x_train={ds_x_tr.shape}  y_train={ds_y_tr.shape}  n_classes={len(np.unique(ds_y_tr))}")
        if ds_y_te is not None:
            print(f"    x_test={ds_x_te.shape if ds_x_te is not None else 'N/A'}  y_test={ds_y_te.shape if ds_y_te is not None else 'N/A'}")

print(f"\n{'='*60}")
print("AUDIT COMPLETE")
print(f"{'='*60}")
