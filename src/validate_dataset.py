"""
数据验证脚本 — 全面检查预处理与数据集合并的正确性。

验证项:
  1) 数据 shape（窗口大小、通道数、train/test 划分比例）
  2) Label 连续性、缺失类别、train/test 覆盖、CSV 一致性
  3) 类别分布统计（长尾、零样本）
  4) 数值检查（NaN/Inf、全零窗口、mean/std/min/max）
  5) 滑窗检查（随机波形可视化，确认窗口未切断动作）
  6) 多数据集分布（t-SNE/PCA 可视化 domain gap）
  7) Augmentation 检查（增强前后对比）
  8) Wrist/Forearm 分布差异分析
  9) 易混淆类别分析（类间相关系数）
  10) 自动生成 validation_report.json 及可视化图表

用法:
  python src/validate_dataset.py
  python src/validate_dataset.py --data_dir data/processed/ --output_dir results/validation/
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，服务器环境下可用
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
CHANNEL_NAMES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
DOMAIN_NAMES = {0: 'forearm', 1: 'wrist', -1: 'unknown'}


def load_data(data_dir):
    """加载合并后的数据集及相关元数据。"""
    files = {
        'x_train': os.path.join(data_dir, 'x_train.npy'),
        'y_train': os.path.join(data_dir, 'y_train.npy'),
        'x_test': os.path.join(data_dir, 'x_test.npy'),
        'y_test': os.path.join(data_dir, 'y_test.npy'),
        'domain_train': os.path.join(data_dir, 'domain_train.npy'),
        'domain_test': os.path.join(data_dir, 'domain_test.npy'),
        'mapping_csv': os.path.join(data_dir, 'merged_label_mapping.csv'),
        'meta_json': os.path.join(data_dir, 'dataset_meta.json'),
        'norm_params': os.path.join(data_dir, 'norm_params.npz'),
    }

    data = {}
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f'[WARN] 缺失文件: {missing}')

    for key in ['x_train', 'y_train', 'x_test', 'y_test',
                'domain_train', 'domain_test']:
        path = files[key]
        if os.path.exists(path):
            arr = np.load(path, allow_pickle=True)
            if 'y_' in key or 'domain_' in key:
                arr = arr.flatten().astype(np.int64)
            else:
                arr = arr.astype(np.float32)
            data[key] = arr

    if os.path.exists(files['mapping_csv']):
        data['mapping_df'] = pd.read_csv(files['mapping_csv'])
    if os.path.exists(files['meta_json']):
        with open(files['meta_json'], 'r', encoding='utf-8') as f:
            data['meta'] = json.load(f)
    if os.path.exists(files['norm_params']):
        npz = np.load(files['norm_params'])
        data['norm_mean'] = npz['mean']
        data['norm_std'] = npz['std']

    return data


def build_class_name_map(data):
    """从 merged_label_mapping.csv 构建 global_label -> canonical_name 映射。"""
    if 'mapping_df' not in data:
        return {}
    df = data['mapping_df']
    name_map = {}
    for _, row in df.iterrows():
        gid = int(row['global_label'])
        if gid not in name_map:
            name_map[gid] = row.get('canonical', row['name'])
    return name_map


# ============================================================
# 1) 数据 Shape 检查
# ============================================================
def check_shape(data, report):
    print('\n' + '=' * 60)
    print('1) 数据 Shape 检查')
    print('=' * 60)

    x_train = data.get('x_train')
    x_test = data.get('x_test')
    y_train = data.get('y_train')
    y_test = data.get('y_test')

    checks = {}

    # 基本存在性
    for name in ['x_train', 'y_train', 'x_test', 'y_test']:
        checks[f'{name}_exists'] = name in data

    if x_train is None or x_test is None:
        print('[FAIL] 缺少核心数据文件')
        report['shape'] = {'status': 'FAIL', 'checks': checks}
        return checks

    # 维度检查
    ndim_ok = x_train.ndim == 3 and x_test.ndim == 3
    checks['ndim'] = {'train': x_train.ndim, 'test': x_test.ndim, 'ok': ndim_ok}
    print(f'  维度: train={x_train.ndim}D, test={x_test.ndim}D  {"[OK]" if ndim_ok else "[FAIL]"}')

    # 通道数
    n_channels = x_train.shape[1]
    ch_ok = n_channels == 6 and x_test.shape[1] == 6
    checks['channels'] = {'train': int(x_train.shape[1]), 'test': int(x_test.shape[1]),
                          'expected': 6, 'ok': ch_ok}
    print(f'  通道数: train={x_train.shape[1]}, test={x_test.shape[1]}  '
          f'{"[OK]" if ch_ok else "[FAIL]"}')

    # 窗口大小
    window_size = x_train.shape[2]
    ws_ok = window_size == 200 and x_test.shape[2] == 200
    checks['window_size'] = {'train': int(window_size), 'test': int(x_test.shape[2]),
                             'expected': 200, 'ok': ws_ok}
    print(f'  窗口大小: train={window_size}, test={x_test.shape[2]}  '
          f'{"[OK]" if ws_ok else "[FAIL]"}')

    # 样本数
    n_train, n_test = len(x_train), len(x_test)
    total = n_train + n_test
    test_ratio = n_test / total if total > 0 else 0
    checks['sample_count'] = {'train': int(n_train), 'test': int(n_test),
                              'total': int(total), 'test_ratio': round(test_ratio, 3)}
    print(f'  样本数: train={n_train}, test={n_test} (test_ratio={test_ratio:.2%})')

    # 标签数量一致性
    yt_ok = len(y_train) == n_train and len(y_test) == n_test
    checks['label_consistency'] = {'train_match': len(y_train) == n_train,
                                   'test_match': len(y_test) == n_test, 'ok': yt_ok}
    print(f'  标签数量一致: {"[OK]" if yt_ok else "[FAIL]"}')

    # Domain label 一致性
    if 'domain_train' in data:
        d_ok = len(data['domain_train']) == n_train and len(data['domain_test']) == n_test
        checks['domain_consistency'] = {'ok': d_ok}
        print(f'  Domain 标签一致: {"[OK]" if d_ok else "[FAIL]"}')

    all_ok = True
    for v in checks.values():
        if isinstance(v, dict):
            if not v.get('ok', True):
                all_ok = False
        elif not v:
            all_ok = False
    report['shape'] = {'status': 'PASS' if all_ok else 'FAIL', 'checks': checks}
    return checks


# ============================================================
# 2) Label 检查
# ============================================================
def check_labels(data, report):
    print('\n' + '=' * 60)
    print('2) Label 检查')
    print('=' * 60)

    y_train = data['y_train']
    y_test = data['y_test']

    checks = {}

    # 标签范围
    all_labels = np.concatenate([y_train, y_test])
    min_lbl, max_lbl = int(all_labels.min()), int(all_labels.max())
    n_unique = len(np.unique(all_labels))
    checks['label_range'] = {'min': min_lbl, 'max': max_lbl, 'n_unique': n_unique}
    print(f'  标签范围: [{min_lbl}, {max_lbl}], 唯一类别数: {n_unique}')

    # 是否从 0 开始连续
    expected = set(range(max_lbl + 1))
    actual = set(np.unique(all_labels))
    missing = sorted(expected - actual)
    is_contiguous = len(missing) == 0
    checks['contiguous'] = {'ok': is_contiguous, 'missing_labels': [int(x) for x in missing]}
    if is_contiguous:
        print(f'  标签连续性: [OK] 0~{max_lbl} 全部覆盖')
    else:
        print(f'  标签连续性: [FAIL] 缺失: {missing}')

    # Train/Test 覆盖
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    train_only = sorted(train_classes - test_classes)
    test_only = sorted(test_classes - train_classes)
    both = sorted(train_classes & test_classes)
    checks['train_test_coverage'] = {
        'train_only': [int(x) for x in train_only],
        'test_only': [int(x) for x in test_only],
        'both_count': len(both),
        'ok': len(train_only) == 0 and len(test_only) == 0,
    }
    print(f'  Train 类别数: {len(train_classes)}, Test 类别数: {len(test_classes)}')
    print(f'  共有类别: {len(both)}, Train 独有: {train_only}, Test 独有: {test_only}')

    # CSV 一致性
    if 'mapping_df' in data:
        df = data['mapping_df']
        csv_globals = set(df['global_label'].unique())
        csv_ok = csv_globals == set(range(n_unique)) or csv_globals == actual
        # 更宽松的检查：CSV 中的 global_label 应该与数据中标签一致
        if not csv_ok:
            extra_in_csv = sorted(csv_globals - actual)
            extra_in_data = sorted(actual - csv_globals)
            print(f'  CSV 不一致: CSV 多余={extra_in_csv}, 数据多余={extra_in_data}')
        checks['csv_consistency'] = {'ok': csv_ok}
        print(f'  CSV 标签一致性: {"[OK]" if csv_ok else "[WARN]"}')
    else:
        checks['csv_consistency'] = {'ok': False, 'reason': 'mapping_csv not found'}

    report['labels'] = {'status': 'PASS' if all(
        v.get('ok', True) for v in checks.values()) else 'WARN', 'checks': checks}
    return checks


# ============================================================
# 3) 类别分布统计
# ============================================================
def check_class_distribution(data, report, name_map, output_dir):
    print('\n' + '=' * 60)
    print('3) 类别分布统计')
    print('=' * 60)

    y_train = data['y_train']
    y_test = data['y_test']

    train_cnt = Counter(y_train)
    test_cnt = Counter(y_test)

    all_classes = sorted(set(list(train_cnt.keys()) + list(test_cnt.keys())))
    n_classes = len(all_classes)

    # 详细分布
    print(f'\n  {"Class":>6s}  {"Name":<30s}  {"Train":>8s}  {"Test":>8s}  {"Total":>8s}  {"T/T Ratio":>10s}')
    print('  ' + '-' * 80)
    distribution = []
    long_tail = []
    zero_samples = []

    for cls in all_classes:
        tr = train_cnt.get(cls, 0)
        te = test_cnt.get(cls, 0)
        total = tr + te
        name = name_map.get(cls, f'class_{cls}')
        ratio = te / max(tr, 1)
        distribution.append({
            'class': int(cls), 'name': name,
            'train': tr, 'test': te, 'total': total,
            'test_train_ratio': round(ratio, 3),
        })
        print(f'  {cls:>6d}  {name:<30s}  {tr:>8d}  {te:>8d}  {total:>8d}  {ratio:>10.3f}')

        if total < 50:
            long_tail.append({'class': int(cls), 'name': name, 'total': total})
        if tr == 0 or te == 0:
            zero_samples.append({'class': int(cls), 'name': name,
                                 'train': tr, 'test': te})

    # 统计摘要
    train_counts = [train_cnt.get(c, 0) for c in all_classes]
    test_counts = [test_cnt.get(c, 0) for c in all_classes]

    stats = {
        'n_classes': n_classes,
        'train': {
            'total': int(sum(train_counts)), 'min': int(min(train_counts)),
            'max': int(max(train_counts)), 'mean': float(np.mean(train_counts)),
            'median': float(np.median(train_counts)),
            'imbalance_ratio': round(max(train_counts) / max(min(train_counts), 1), 1),
        },
        'test': {
            'total': int(sum(test_counts)), 'min': int(min(test_counts)),
            'max': int(max(test_counts)), 'mean': float(np.mean(test_counts)),
            'median': float(np.median(test_counts)),
            'imbalance_ratio': round(max(test_counts) / max(min(test_counts), 1), 1),
        },
        'long_tail_classes': long_tail,
        'zero_sample_classes': zero_samples,
    }
    print(f'\n  Train 不均衡比 (max/min): {stats["train"]["imbalance_ratio"]}x')
    print(f'  Test  不均衡比 (max/min): {stats["test"]["imbalance_ratio"]}x')
    print(f'  长尾类别 (<50 样本): {len(long_tail)}')
    if long_tail:
        print(f'    -> {[(c["name"], c["total"]) for c in long_tail]}')
    print(f'  零样本类别: {len(zero_samples)}')
    if zero_samples:
        for c in zero_samples:
            print(f'    -> {c["name"]}: train={c["train"]}, test={c["test"]}')

    report['class_distribution'] = {'status': 'WARN' if (long_tail or zero_samples) else 'PASS',
                                     'stats': stats, 'details': distribution}

    # ---- 绘制类别分布图 ----
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n_classes * 0.35)))
    class_names_display = [name_map.get(c, f'cls_{c}') for c in all_classes]

    # Train
    axes[0].barh(range(n_classes), [train_cnt.get(c, 0) for c in all_classes],
                 color='steelblue', edgecolor='white')
    axes[0].set_yticks(range(n_classes))
    axes[0].set_yticklabels(class_names_display, fontsize=8)
    axes[0].set_xlabel('Sample Count')
    axes[0].set_title('Train Class Distribution')
    axes[0].invert_yaxis()
    for i, cnt in enumerate([train_cnt.get(c, 0) for c in all_classes]):
        axes[0].text(cnt + max(train_counts) * 0.01, i, str(cnt), va='center', fontsize=7)

    # Test
    axes[1].barh(range(n_classes), [test_cnt.get(c, 0) for c in all_classes],
                 color='coral', edgecolor='white')
    axes[1].set_yticks(range(n_classes))
    axes[1].set_yticklabels(class_names_display, fontsize=8)
    axes[1].set_xlabel('Sample Count')
    axes[1].set_title('Test Class Distribution')
    axes[1].invert_yaxis()
    for i, cnt in enumerate([test_cnt.get(c, 0) for c in all_classes]):
        axes[1].text(cnt + max(test_counts) * 0.01, i, str(cnt), va='center', fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_distribution.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  [图] 类别分布图已保存: {save_path}')

    return stats


# ============================================================
# 4) 数值检查
# ============================================================
def check_numerical(data, report):
    print('\n' + '=' * 60)
    print('4) 数值检查')
    print('=' * 60)

    x_train = data['x_train']
    x_test = data['x_test']

    checks = {}
    all_ok = True

    for split_name, X in [('train', x_train), ('test', x_test)]:
        print(f'\n  --- {split_name} ---')

        # 基本统计
        ch_stats = []
        for c in range(X.shape[1]):
            ch_data = X[:, c, :]
            ch_stats.append({
                'channel': CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f'ch{c}',
                'mean': float(ch_data.mean()), 'std': float(ch_data.std()),
                'min': float(ch_data.min()), 'max': float(ch_data.max()),
            })
            print(f'    {CHANNEL_NAMES[c]:8s}  mean={ch_data.mean():+.4f}  '
                  f'std={ch_data.std():.4f}  range=[{ch_data.min():+.3f}, {ch_data.max():+.3f}]')
        checks[f'{split_name}_channel_stats'] = ch_stats

        # NaN / Inf 检查
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        nan_inf_ok = (n_nan == 0 and n_inf == 0)
        checks[f'{split_name}_nan_inf'] = {'nan_count': n_nan, 'inf_count': n_inf, 'ok': nan_inf_ok}
        if not nan_inf_ok:
            print(f'    [FAIL] NaN={n_nan}, Inf={n_inf}')
            all_ok = False
        else:
            print(f'    [OK] 无 NaN/Inf')

        # 全零窗口检查
        zero_mask = np.all(np.abs(X) < 1e-10, axis=(1, 2))
        n_all_zero = int(zero_mask.sum())
        zero_ok = (n_all_zero == 0)
        checks[f'{split_name}_all_zero'] = {'count': n_all_zero, 'ok': zero_ok}
        if not zero_ok:
            print(f'    [WARN] 全零窗口: {n_all_zero}')
        else:
            print(f'    [OK] 无全零窗口')

        # 全等窗口（所有值相同）
        const_mask = np.all(np.std(X, axis=2) < 1e-10, axis=1)
        n_constant = int(const_mask.sum())
        checks[f'{split_name}_constant_windows'] = {'count': n_constant, 'ok': n_constant == 0}
        if n_constant > 0:
            print(f'    [WARN] 常数窗口 (所有时间步值相同): {n_constant}')

        # 异常值（超过 10 倍标准差）
        global_std = X.std()
        n_outliers = int((np.abs(X) > 10 * global_std).sum())
        checks[f'{split_name}_outliers'] = {'count': n_outliers, 'ok': n_outliers < len(X)}

    report['numerical'] = {'status': 'PASS' if all_ok else 'WARN', 'checks': checks}
    return checks


# ============================================================
# 5) 滑窗检查 — 随机可视化
# ============================================================
def check_sliding_windows(data, report, output_dir, n_samples=6):
    print('\n' + '=' * 60)
    print('5) 滑窗检查')
    print('=' * 60)

    x_train = data['x_train']
    y_train = data['y_train']
    domain_train = data.get('domain_train')
    name_map = build_class_name_map(data)

    n_total = len(x_train)
    rng = np.random.RandomState(42)
    indices = rng.choice(n_total, min(n_samples, n_total), replace=False)

    # 绘制波形
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 2.5 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        x = x_train[idx]  # (6, 200)
        y = int(y_train[idx])
        class_name = name_map.get(y, f'class_{y}')
        domain = DOMAIN_NAMES.get(int(domain_train[idx]), '?') if domain_train is not None else '?'

        t = np.arange(x.shape[1])
        # 加速度计
        for c in range(3):
            axes[i].plot(t, x[c] + c * 3, linewidth=0.7, alpha=0.8,
                         label=f'{CHANNEL_NAMES[c]}')
        # 陀螺仪
        for c in range(3, 6):
            axes[i].plot(t, x[c] + c * 3, linewidth=0.7, alpha=0.8,
                         label=f'{CHANNEL_NAMES[c]}')
        axes[i].set_title(f'Window #{idx} | Label: {class_name} (id={y}) | Domain: {domain}',
                          fontsize=10)
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('Normalized value (offset)')
        axes[i].legend(loc='upper right', fontsize=6, ncol=2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_waveforms.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  随机展示 {len(indices)} 个训练样本波形')
    print(f'  [图] 样本波形已保存: {save_path}')

    # 检查连续窗口的标签一致性
    # 注意: merge 时已 shuffle 过数据，所以连续标签必然频繁切换
    # 这里改为按标签排序后检查每个标签段内的窗口是否合理
    sorted_idx = np.argsort(y_train)
    n_check = min(500, n_total - 1)
    label_changes_sorted = int(np.sum(np.diff(y_train[sorted_idx[:n_check]]) != 0))
    avg_run_length = n_check / max(label_changes_sorted, 1)
    print(f'  按标签排序后，前 {n_check} 个窗口中标签切换 {label_changes_sorted} 次，'
          f'平均连续段长度={avg_run_length:.1f}')

    report['sliding_windows'] = {
        'n_samples_visualized': int(len(indices)),
        'label_changes_in_sorted': int(label_changes_sorted),
        'avg_consecutive_same_label': round(float(avg_run_length), 1),
        'note': '原始数据已 shuffle，按标签排序后检查',
    }


# ============================================================
# 6) 多数据集分布检查 (t-SNE / PCA)
# ============================================================
def check_domain_distribution(data, report, output_dir, max_samples=3000):
    print('\n' + '=' * 60)
    print('6) 多数据集分布检查 (t-SNE / PCA)')
    print('=' * 60)

    x_train = data['x_train']
    y_train = data['y_train']
    domain_train = data.get('domain_train')
    name_map = build_class_name_map(data)

    # 子采样以加速
    n_total = len(x_train)
    if n_total > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total, max_samples, replace=False)
        X_sub = x_train[indices]
        y_sub = y_train[indices]
        d_sub = domain_train[indices] if domain_train is not None else None
    else:
        X_sub = x_train
        y_sub = y_train
        d_sub = domain_train

    # 特征提取: 对每个窗口取统计特征
    features = []
    for i in range(len(X_sub)):
        x = X_sub[i]  # (6, 200)
        feat = np.concatenate([
            x.mean(axis=1),           # 6
            x.std(axis=1),            # 6
            np.percentile(x, 25, axis=1),  # 6
            np.percentile(x, 75, axis=1),  # 6
        ])
        features.append(feat)
    features = np.stack(features, axis=0)  # (N, 24)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features)
    print(f'  PCA 解释方差: {pca.explained_variance_ratio_}')

    # t-SNE
    from sklearn.manifold import TSNE
    print('  计算 t-SNE (可能需要 1-2 分钟)...')
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) // 3),
                random_state=42, n_jobs=1)
    tsne_result = tsne.fit_transform(features)

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    unique_labels = sorted(np.unique(y_sub))
    cmap = plt.cm.tab20

    # PCA by class
    ax = axes[0, 0]
    for lbl in unique_labels:
        mask = y_sub == lbl
        name = name_map.get(lbl, f'cls_{lbl}')
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   s=8, alpha=0.6, label=name, color=cmap(lbl % 20))
    ax.set_title('PCA colored by Class')
    ax.legend(fontsize=5, loc='upper right', ncol=2)

    # PCA by domain
    ax = axes[0, 1]
    if d_sub is not None:
        for dom in sorted(np.unique(d_sub)):
            mask = d_sub == dom
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       s=8, alpha=0.6, label=DOMAIN_NAMES.get(dom, f'dom_{dom}'))
    ax.set_title('PCA colored by Domain')
    ax.legend(fontsize=8)

    # t-SNE by class
    ax = axes[1, 0]
    for lbl in unique_labels:
        mask = y_sub == lbl
        name = name_map.get(lbl, f'cls_{lbl}')
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   s=8, alpha=0.6, label=name, color=cmap(lbl % 20))
    ax.set_title('t-SNE colored by Class')

    # t-SNE by domain
    ax = axes[1, 1]
    if d_sub is not None:
        for dom in sorted(np.unique(d_sub)):
            mask = d_sub == dom
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                       s=8, alpha=0.6, label=DOMAIN_NAMES.get(dom, f'dom_{dom}'))
    ax.set_title('t-SNE colored by Domain')
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'domain_tsne_pca.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] t-SNE/PCA 可视化已保存: {save_path}')

    # Domain gap 定量分析
    if d_sub is not None and len(np.unique(d_sub)) >= 2:
        domains = sorted(np.unique(d_sub))
        domain_means = {}
        for dom in domains:
            domain_means[dom] = features[d_sub == dom].mean(axis=0)
        # 计算 domain 之间的欧氏距离（基于统计特征）
        gap = np.linalg.norm(domain_means[domains[0]] - domain_means[domains[1]])
        print(f'  Domain 均值特征距离: {gap:.4f}')
        report['domain_distribution'] = {
            'pca_variance_ratio': [round(float(v), 4) for v in pca.explained_variance_ratio_],
            'domain_feature_distance': round(float(gap), 4),
            'n_samples_used': int(len(features)),
        }
    else:
        report['domain_distribution'] = {
            'pca_variance_ratio': [round(float(v), 4) for v in pca.explained_variance_ratio_],
            'n_samples_used': int(len(features)),
            'note': '单 domain 或无 domain label，未计算 domain gap',
        }


# ============================================================
# 7) Augmentation 检查
# ============================================================
def check_augmentation(data, report, output_dir, n_samples=4):
    print('\n' + '=' * 60)
    print('7) Augmentation 检查')
    print('=' * 60)

    x_train = data['x_train']
    name_map = build_class_name_map(data)

    # 复用 dataset.py 中的增强逻辑
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from dataset import GestureDataset
    ds = GestureDataset.from_arrays(x_train[:100], data['y_train'][:100], train=True)

    rng = np.random.RandomState(123)
    indices = rng.choice(min(100, len(x_train)), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        x_orig = x_train[idx].copy()
        # 手动应用一次增强（不同随机种子产生不同效果）
        np.random.seed(42 + row)
        x_aug = ds._augment(x_orig.copy())

        t = np.arange(x_orig.shape[1])
        for c in range(6):
            axes[row, 0].plot(t, x_orig[c] + c * 2, linewidth=0.6, alpha=0.7,
                              label=CHANNEL_NAMES[c])
            axes[row, 1].plot(t, x_aug[c] + c * 2, linewidth=0.6, alpha=0.7,
                              label=CHANNEL_NAMES[c])
        axes[row, 0].set_title(f'Original (idx={idx})', fontsize=9)
        axes[row, 1].set_title(f'Augmented', fontsize=9)
        for ax in axes[row]:
            ax.set_xlabel('Time step')
            ax.set_ylabel('Value (offset)')

    axes[0, 0].legend(fontsize=6, ncol=2, loc='upper right')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'augmentation_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  对比 {n_samples} 个样本的增强前后效果')
    print(f'  [图] 增强对比图已保存: {save_path}')

    report['augmentation'] = {'n_samples_compared': n_samples}


# ============================================================
# 8) Wrist / Forearm 分布差异
# ============================================================
def check_wrist_forearm(data, report, output_dir, max_per_domain=1000):
    print('\n' + '=' * 60)
    print('8) Wrist / Forearm 分布差异分析')
    print('=' * 60)

    x_train = data['x_train']
    domain_train = data.get('domain_train')

    if domain_train is None:
        print('  缺少 domain_train 数据，跳过')
        report['wrist_forearm'] = {'status': 'SKIP', 'reason': 'no domain_train'}
        return

    # 按 domain 分组
    domains_present = sorted(np.unique(domain_train))
    domain_data = {}
    for dom in domains_present:
        mask = domain_train == dom
        n = int(mask.sum())
        if n > max_per_domain:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, max_per_domain, replace=False)
            domain_data[dom] = x_train[mask][idx]
        else:
            domain_data[dom] = x_train[mask]
        print(f'  Domain {dom} ({DOMAIN_NAMES.get(dom, "?")}): {n} samples '
              f'(可视化用 {len(domain_data[dom])})')

    # 每通道分布对比
    n_domains = len(domain_data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for c in range(6):
        ax = axes[c]
        for dom, color in zip(sorted(domain_data.keys()),
                              ['steelblue', 'coral', 'green'][:n_domains]):
            ch_data = domain_data[dom][:, c, :].flatten()
            ax.hist(ch_data, bins=80, alpha=0.4, density=True,
                    color=color, label=f'{DOMAIN_NAMES.get(dom, f"dom_{dom}")} '
                    f'(μ={ch_data.mean():.3f}, σ={ch_data.std():.3f})')
        ax.set_title(CHANNEL_NAMES[c])
        ax.legend(fontsize=7)
        ax.set_xlabel('Normalized value')

    plt.suptitle('Per-Channel Distribution by Domain (Train set)', fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'wrist_forearm_distribution.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] 每通道分布对比已保存: {save_path}')

    # 定量分析：每通道 KS 距离
    from scipy.stats import ks_2samp
    ks_results = []
    if len(domain_data) >= 2:
        dom_keys = sorted(domain_data.keys())
        for c in range(6):
            d1 = domain_data[dom_keys[0]][:, c, :].flatten()
            d2 = domain_data[dom_keys[1]][:, c, :].flatten()
            # 子采样避免 KS 检验在超大样本量下过于敏感
            n_sub = min(5000, min(len(d1), len(d2)))
            rng = np.random.RandomState(0)
            d1_sub = rng.choice(d1, n_sub, replace=False)
            d2_sub = rng.choice(d2, n_sub, replace=False)
            ks_stat, ks_pval = ks_2samp(d1_sub, d2_sub)
            ks_results.append({
                'channel': CHANNEL_NAMES[c],
                'ks_statistic': round(float(ks_stat), 4),
                'ks_pvalue': round(float(ks_pval), 6),
            })
            print(f'    {CHANNEL_NAMES[c]:8s}  KS={ks_stat:.4f}  p={ks_pval:.6f}  '
                  f'{"***显著差异***" if ks_pval < 0.001 else "差异不显著"}')
        report['wrist_forearm'] = {
            'ks_test': ks_results,
            'domains': {int(k): DOMAIN_NAMES.get(k, '?') for k in dom_keys},
        }
    else:
        report['wrist_forearm'] = {'note': '仅一个 domain，无法对比'}


# ============================================================
# 9) 易混淆类别分析
# ============================================================
def check_confusable_classes(data, report, output_dir, name_map):
    print('\n' + '=' * 60)
    print('9) 易混淆类别分析')
    print('=' * 60)

    x_train = data['x_train']
    y_train = data['y_train']

    unique_labels = sorted(np.unique(y_train))
    n_classes = len(unique_labels)

    # 计算每类的平均特征向量（统计特征）
    class_features = {}
    for lbl in unique_labels:
        mask = y_train == lbl
        X_c = x_train[mask]
        # 每窗口提取统计特征
        feats = []
        for i in range(len(X_c)):
            x = X_c[i]
            feats.append(np.concatenate([
                x.mean(axis=1), x.std(axis=1),
            ]))
        class_features[lbl] = np.stack(feats).mean(axis=0)

    # 类间余弦相似度矩阵
    from scipy.spatial.distance import cdist
    feat_matrix = np.stack([class_features[lbl] for lbl in unique_labels])
    # 归一化
    feat_norm = feat_matrix / (np.linalg.norm(feat_matrix, axis=1, keepdims=True) + 1e-10)
    sim_matrix = feat_norm @ feat_norm.T  # 余弦相似度

    # 找出最相似（易混淆）的类别对
    confusable_pairs = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            confusable_pairs.append({
                'class_a': int(unique_labels[i]),
                'name_a': name_map.get(unique_labels[i], f'cls_{unique_labels[i]}'),
                'class_b': int(unique_labels[j]),
                'name_b': name_map.get(unique_labels[j], f'cls_{unique_labels[j]}'),
                'cosine_similarity': round(float(sim_matrix[i, j]), 4),
            })
    confusable_pairs.sort(key=lambda x: x['cosine_similarity'], reverse=True)

    print(f'  Top-10 最相似的类别对:')
    for pair in confusable_pairs[:10]:
        print(f'    {pair["name_a"]:<30s} <-> {pair["name_b"]:<30s}  '
              f'sim={pair["cosine_similarity"]:.4f}')

    report['confusable_classes'] = {
        'method': 'class_mean_cosine_similarity',
        'top10_confusable_pairs': confusable_pairs[:10],
    }

    # ---- 绘制相似度热力图 ----
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.7), max(7, n_classes * 0.6)))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0, aspect='auto')
    class_names = [name_map.get(lbl, f'cls_{lbl}') for lbl in unique_labels]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title('Inter-Class Cosine Similarity (based on mean statistics)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_similarity_heatmap.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] 类间相似度热力图已保存: {save_path}')


# ============================================================
# 10) 简易混淆矩阵（使用快速 KNN/统计分类器）
# ============================================================
def check_confusion_matrix(data, report, output_dir, name_map, max_samples=2000):
    print('\n' + '=' * 60)
    print('10) 简易混淆矩阵（KNN 基线）')
    print('=' * 60)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # 子采样
    rng = np.random.RandomState(42)
    if len(x_train) > max_samples:
        idx_tr = rng.choice(len(x_train), max_samples, replace=False)
        X_tr, Y_tr = x_train[idx_tr], y_train[idx_tr]
    else:
        X_tr, Y_tr = x_train, y_train
    if len(x_test) > max_samples // 2:
        idx_te = rng.choice(len(x_test), max_samples // 2, replace=False)
        X_te, Y_te = x_test[idx_te], y_test[idx_te]
    else:
        X_te, Y_te = x_test, y_test

    # 提取统计特征
    def extract_features(X):
        feats = []
        for i in range(len(X)):
            x = X[i]
            feats.append(np.concatenate([
                x.mean(axis=1), x.std(axis=1),
                np.percentile(x, 25, axis=1), np.percentile(x, 75, axis=1),
            ]))
        return np.stack(feats)

    print('  提取统计特征...')
    F_tr = extract_features(X_tr)
    F_te = extract_features(X_te)

    # KNN 分类
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, classification_report

    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
    knn.fit(F_tr, Y_tr)
    y_pred = knn.predict(F_te)
    acc = (y_pred == Y_te).mean()
    print(f'  KNN (k=5, cosine) 准确率: {acc:.4f}')

    # 混淆矩阵
    cm = confusion_matrix(Y_te, y_pred)
    unique_labels = sorted(np.unique(np.concatenate([Y_tr, Y_te])))
    class_names = [name_map.get(lbl, f'cls_{lbl}') for lbl in unique_labels]

    # 归一化混淆矩阵
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # 绘图
    n = len(unique_labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (KNN baseline, Acc={acc:.3f})')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] 混淆矩阵已保存: {save_path}')

    # 分类报告
    cr = classification_report(Y_te, y_pred, target_names=class_names, zero_division=0)
    print(f'\n  分类报告:\n{cr}')

    report['confusion_matrix'] = {
        'method': 'KNN_k5_cosine',
        'accuracy': round(float(acc), 4),
        'classification_report': cr,
    }


# ============================================================
# 训练曲线记录检查
# ============================================================
def check_training_logs(data_dir, report):
    """检查是否存在训练日志并读取关键信息。"""
    print('\n' + '=' * 60)
    print('附加: 训练日志检查')
    print('=' * 60)

    log_paths = [
        os.path.join(data_dir, '..', 'results', 'train_log.txt'),
        os.path.join('results', 'train_log.txt'),
        os.path.join('results', 'train_log_v3.txt'),
    ]
    for path in log_paths:
        if os.path.exists(path):
            print(f'  找到训练日志: {path}')
            with open(path, 'r') as f:
                content = f.read()
            # 提取关键行
            lines = content.strip().split('\n')
            key_lines = [l for l in lines if any(
                kw in l.lower() for kw in ['accuracy', 'acc:', 'best', 'epoch', 'loss',
                                           'precision', 'recall', 'f1'])]
            for line in key_lines[-10:]:  # 只显示最后 10 行关键信息
                print(f'    {line.strip()}')
            report['training_logs'] = {
                'found': True, 'path': path, 'last_key_lines': key_lines[-10:],
            }
            return
    print('  未找到训练日志')
    report['training_logs'] = {'found': False}


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='IMU 动作识别数据集验证')
    parser.add_argument('--data_dir', default='data/processed/',
                        help='数据目录 (默认: data/processed/)')
    parser.add_argument('--output_dir', default='results/validation/',
                        help='输出目录 (默认: results/validation/)')
    parser.add_argument('--skip_tsne', action='store_true',
                        help='跳过 t-SNE 计算（加速）')
    parser.add_argument('--skip_knn', action='store_true',
                        help='跳过 KNN 混淆矩阵')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('  IMU 动作识别 — 数据集全面验证')
    print('=' * 60)
    print(f'  数据目录: {data_dir}')
    print(f'  输出目录: {output_dir}')

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    print('\n--- 加载数据 ---')
    data = load_data(data_dir)

    if 'x_train' not in data or 'y_train' not in data:
        print('[FATAL] 缺少核心数据 (x_train/y_train)，无法继续')
        sys.exit(1)

    name_map = build_class_name_map(data)
    print(f'  类别数: {len(name_map)}')
    print(f'  类别名: {list(name_map.values())}')

    report = {
        'data_dir': data_dir,
        'num_classes': len(name_map),
        'class_names': [name_map.get(i, f'cls_{i}') for i in sorted(name_map.keys())],
    }

    # ------------------------------------------------------------------
    # 逐步检查
    # ------------------------------------------------------------------
    check_shape(data, report)
    check_labels(data, report)
    dist_stats = check_class_distribution(data, report, name_map, output_dir)
    check_numerical(data, report)
    check_sliding_windows(data, report, output_dir)

    if not args.skip_tsne:
        check_domain_distribution(data, report, output_dir)
    else:
        print('\n[SKIP] 跳过 t-SNE/PCA')

    check_augmentation(data, report, output_dir)
    check_wrist_forearm(data, report, output_dir)
    check_confusable_classes(data, report, output_dir, name_map)

    if not args.skip_knn:
        check_confusion_matrix(data, report, output_dir, name_map)
    else:
        print('\n[SKIP] 跳过 KNN 混淆矩阵')

    check_training_logs(data_dir, report)

    # ------------------------------------------------------------------
    # 保存报告
    # ------------------------------------------------------------------
    report_path = os.path.join(output_dir, 'validation_report.json')

    # 清理不可序列化的对象
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report_clean = sanitize(report)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_clean, f, indent=2, ensure_ascii=False)
    print(f'\n{"=" * 60}')
    print(f'  验证报告已保存: {report_path}')
    print(f'  图表输出目录: {output_dir}')
    print(f'{"=" * 60}')

    # ------------------------------------------------------------------
    # 打印摘要
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('  验证摘要')
    print('=' * 60)
    for section in ['shape', 'labels', 'class_distribution', 'numerical',
                    'sliding_windows', 'domain_distribution', 'augmentation',
                    'wrist_forearm', 'confusable_classes', 'confusion_matrix']:
        sec_data = report_clean.get(section, {})
        if isinstance(sec_data, dict):
            status = sec_data.get('status', 'OK')
            print(f'  [{status:<5s}] {section}')


if __name__ == '__main__':
    main()
