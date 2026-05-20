"""
特征嵌入与时序注意力可视化。

功能:
  1. t-SNE / UMAP 特征嵌入可视化 — 分析动作分离程度
  2. 时序注意力热力图 — 查看模型关注哪些时间段
  3. 每类嵌入空间分析 — 类内紧凑度 / 类间分离度
  4. Domain 分布可视化 — 检查 domain gap 是否缩小

用法:
  # 基础: 可视化已训练模型的特征嵌入
  python -X utf8 src/visualization.py --model TCN --features results/features_TCN.npy

  # 完整: 从模型直接生成所有可视化
  python -X utf8 src/visualization.py --model TCN --ckpt models/best_TCN_v3.pth \\
      --data_dir data/processed/ --all

  # 仅时序注意力
  python -X utf8 src/visualization.py --model AttnConvLSTM \\
      --ckpt models/best_AttnConvLSTM_v3.pth --attention
"""

import torch
import numpy as np
import os, sys, json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v3 import create_model_v3
from model_v2 import create_model_v2
from dataset import GestureDataset

RESULT_DIR = 'results/visualization/'
os.makedirs(RESULT_DIR, exist_ok=True)

CHANNEL_NAMES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']


# ============================================================
# 1) t-SNE 特征嵌入可视化
# ============================================================
def visualize_tsne(features, labels, class_names, title='Feature Embedding (t-SNE)',
                   save_path=None, max_samples=2000, perplexity=30):
    """t-SNE 降维到 2D 并可视化。

    输出:
      - 散点图 (按类别着色)
      - 每类中心标注
      - 轮廓系数 (silhouette score)
    """
    print(f"\n{'='*60}")
    print(f"t-SNE 特征嵌入可视化")
    print(f"{'='*60}")

    # 子采样
    if len(features) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    print(f"  样本数: {len(features)}, 特征维度: {features.shape[1]}")

    # 归一化
    from sklearn.preprocessing import StandardScaler
    features_norm = StandardScaler().fit_transform(features)

    # t-SNE
    print(f"  计算 t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features) // 3),
                random_state=42, n_jobs=1, learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(features_norm)

    # PCA 对照
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_norm)

    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)
    cmap = plt.cm.tab20

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # t-SNE
    ax = axes[0]
    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else f'cls_{lbl}'
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   s=10, alpha=0.6, label=name, color=cmap(lbl % 20))

        # 标注类中心
        center = tsne_result[mask].mean(axis=0)
        ax.annotate(name[:12], center, fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_title(f't-SNE: {title}\n(PCA var: {pca.explained_variance_ratio_[0]:.1%}'
                 f'+{pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.legend(fontsize=5, loc='upper right', ncol=2)

    # PCA
    ax = axes[1]
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   s=10, alpha=0.6, color=cmap(lbl % 20))
    ax.set_title('PCA (baseline)', fontsize=10)

    plt.tight_layout()
    sp = save_path or os.path.join(RESULT_DIR, 'tsne_embedding.png')
    fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] t-SNE 已保存: {sp}")

    # ---- 定量分析 ----
    try:
        sil_score = silhouette_score(features_norm, labels, sample_size=min(2000, len(features)))
        print(f"  轮廓系数 (silhouette): {sil_score:.4f} (越高越好，max=1.0)")
    except Exception:
        sil_score = None
        print(f"  轮廓系数: 无法计算 (需要每类别至少2个样本)")

    # 类内紧凑度 / 类间分离度
    class_centers = {}
    for lbl in unique_labels:
        class_centers[lbl] = features_norm[labels == lbl].mean(axis=0)

    intra_dists = []
    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() > 1:
            center = class_centers[lbl]
            dists = np.linalg.norm(features_norm[mask] - center, axis=1)
            intra_dists.append(dists.mean())
    avg_intra = np.mean(intra_dists) if intra_dists else 0

    inter_dists = []
    centers_mat = np.stack(list(class_centers.values()))
    for i in range(len(centers_mat)):
        for j in range(i + 1, len(centers_mat)):
            inter_dists.append(np.linalg.norm(centers_mat[i] - centers_mat[j]))
    avg_inter = np.mean(inter_dists) if inter_dists else 0

    sep_ratio = avg_inter / max(avg_intra, 1e-10)
    print(f"  类内平均距离: {avg_intra:.4f}  (越小越好)")
    print(f"  类间平均距离: {avg_inter:.4f}  (越大越好)")
    print(f"  分离比 (inter/intra): {sep_ratio:.2f}  (越大越好)")

    return {
        'silhouette_score': float(sil_score) if sil_score else None,
        'intra_class_dist': float(avg_intra),
        'inter_class_dist': float(avg_inter),
        'separation_ratio': float(sep_ratio),
    }


# ============================================================
# 2) UMAP 可视化 (需要 umap-learn)
# ============================================================
def visualize_umap(features, labels, class_names, title='Feature Embedding (UMAP)',
                   save_path=None, max_samples=2000):
    """UMAP 降维可视化 — 比 t-SNE 更好地保留全局结构。"""
    try:
        import umap
    except ImportError:
        print("  [SKIP] umap-learn 未安装 (pip install umap-learn)")
        return

    print(f"\n{'='*60}")
    print(f"UMAP 特征嵌入可视化")
    print(f"{'='*60}")

    if len(features) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    from sklearn.preprocessing import StandardScaler
    features_norm = StandardScaler().fit_transform(features)

    print(f"  计算 UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    umap_result = reducer.fit_transform(features_norm)

    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(12, 10))
    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else f'cls_{lbl}'
        ax.scatter(umap_result[mask, 0], umap_result[mask, 1],
                   s=10, alpha=0.6, label=name, color=cmap(lbl % 20))
    ax.set_title(f'UMAP: {title}', fontsize=12)
    ax.legend(fontsize=6, loc='upper right', ncol=2)

    plt.tight_layout()
    sp = save_path or os.path.join(RESULT_DIR, 'umap_embedding.png')
    fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] UMAP 已保存: {sp}")


# ============================================================
# 3) 时序注意力可视化
# ============================================================
def visualize_temporal_attention(model, dataloader, device, class_names,
                                 save_path=None, n_samples=8):
    """可视化模型对时间步的注意力权重。

    选择多种动作类别样本，显示模型重点关注的时间段。
    帮助理解:
      - 模型是否在学习完整动作周期
      - 是否仅依赖局部统计特征
    """
    print(f"\n{'='*60}")
    print(f"时序注意力可视化")
    print(f"{'='*60}")

    model.eval()
    samples = []
    target_classes = set()

    # 收集不同类别的样本
    for x_batch, y_batch in dataloader:
        for i in range(len(x_batch)):
            lbl = int(y_batch[i])
            if lbl not in target_classes and len(samples) < n_samples:
                target_classes.add(lbl)
                samples.append((x_batch[i:i+1], lbl, len(samples)))
        if len(samples) >= n_samples:
            break

    if not samples:
        print("  无法收集样本")
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, (x, lbl, idx) in enumerate(samples):
        x = x.to(device)
        cls_name = class_names[lbl] if lbl < len(class_names) else f'cls_{lbl}'

        with torch.no_grad():
            try:
                logits, features, attn_weights = model(x, return_features=True)
            except Exception:
                # 模型可能不支持 return_attn
                print(f"  模型不支持 return_features=True 或注意力权重")
                plt.close(fig)
                return

        x_np = x.cpu().numpy()[0]  # (6, 200)
        pred = logits.argmax(1).item()
        pred_name = class_names[pred] if pred < len(class_names) else f'cls_{pred}'

        # 左: IMU 波形
        t = np.arange(x_np.shape[1])
        for c in range(6):
            axes[row, 0].plot(t, x_np[c] + c * 2, linewidth=0.6, alpha=0.7,
                              label=CHANNEL_NAMES[c])
        axes[row, 0].set_title(f'Waveform (True: {cls_name}, Pred: {pred_name})',
                               fontsize=8)
        axes[row, 0].set_xlabel('Time step')
        axes[row, 0].set_ylabel('Value (offset)')
        if row == 0:
            axes[row, 0].legend(fontsize=5, ncol=2)

        # 右: 注意力权重 (如果可用)
        if isinstance(attn_weights, dict):
            # MultiStreamCNN: 三流注意力
            for stream_name, w in attn_weights.items():
                w_np = w.cpu().numpy()[0]  # (T_stream,)
                # 上采样到 200
                t_stream = np.linspace(0, 199, len(w_np))
                axes[row, 1].plot(t_stream, w_np, linewidth=1,
                                  label=stream_name, alpha=0.7)
            axes[row, 1].set_title(f'Multi-stream attention weights')
            axes[row, 1].legend(fontsize=7)
        elif attn_weights is not None and attn_weights.ndim >= 1:
            # 单流注意力: (B, T)
            w_np = attn_weights.cpu().numpy()
            if w_np.ndim == 2:
                w_np = w_np[0]
            t_attn = np.linspace(0, 199, len(w_np))
            axes[row, 1].fill_between(t_attn, 0, w_np, alpha=0.4, color='steelblue')
            axes[row, 1].plot(t_attn, w_np, linewidth=1, color='steelblue')
            axes[row, 1].set_title(f'Temporal Attention (peak at t={np.argmax(w_np)})')
            axes[row, 1].set_ylabel('Attention weight')
            axes[row, 1].set_xlabel('Effective time step')
        else:
            axes[row, 1].text(0.5, 0.5, 'No attention available',
                              ha='center', va='center', transform=axes[row, 1].transAxes)

    plt.tight_layout()
    sp = save_path or os.path.join(RESULT_DIR, 'temporal_attention.png')
    fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 时序注意力已保存: {sp}")


# ============================================================
# 4) Domain Gap 可视化
# ============================================================
def visualize_domain_gap(features, labels, domain_labels, class_names,
                         save_path=None):
    """可视化特征空间中不同 domain 的分布差异。

    同一类别在不同 domain 下特征是否聚集？
    - 如果同一类跨 domain 紧密 → domain-invariant 特征 ✓
    - 如果同一类按 domain 分裂 → domain gap 严重 ✗
    """
    print(f"\n{'='*60}")
    print(f"Domain Gap 可视化")
    print(f"{'='*60}")

    from sklearn.preprocessing import StandardScaler
    features_norm = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_norm)

    # 找出 2 个 domain 共有的类别
    domains = sorted(np.unique(domain_labels))
    if len(domains) < 2:
        print("  仅 1 个 domain，无法分析 gap")
        return

    # 按类别可视化 domain 分布
    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)
    n_cols = min(5, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, lbl in enumerate(unique_labels):
        ax = axes[i]
        cls_mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else f'cls_{lbl}'

        for dom, marker, color in zip(domains, ['o', '^', 's'], ['steelblue', 'coral', 'green']):
            mask = cls_mask & (domain_labels == dom)
            if mask.sum() > 0:
                ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                           s=8, alpha=0.6, marker=marker, color=color,
                           label=f'dom_{dom}' if i == 0 else '')

        ax.set_title(f'{name[:20]} (n={cls_mask.sum()})', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if n_classes > 0:
        axes[0].legend(fontsize=7)

    fig.suptitle('Per-Class Domain Distribution (PCA)', fontsize=12)
    plt.tight_layout()
    sp = save_path or os.path.join(RESULT_DIR, 'domain_gap.png')
    fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] Domain Gap 已保存: {sp}")

    # 定量: 每类的跨 domain 距离
    print(f"\n  跨 Domain 分布分析:")
    for lbl in unique_labels[:10]:  # 只显示前 10 类
        cls_mask = labels == lbl
        dom_0 = features_norm[cls_mask & (domain_labels == domains[0])]
        dom_1 = features_norm[cls_mask & (domain_labels == domains[1])]
        if len(dom_0) > 0 and len(dom_1) > 0:
            center_dist = np.linalg.norm(dom_0.mean(axis=0) - dom_1.mean(axis=0))
            name = class_names[lbl] if lbl < len(class_names) else f'cls_{lbl}'
            marker = ' *** HIGH GAP ***' if center_dist > 2.0 else ''
            print(f"    {name:<30s}  center_dist={center_dist:.3f}{marker}")


# ============================================================
# 5) 混淆类别深入分析
# ============================================================
def analyze_confusable_classes(features, labels, preds, class_names,
                               save_path=None, top_k=5):
    """深入分析最易混淆的类别对。

    对每对混淆类别:
      - 在 2D 空间中可视化
      - 输出定量分离指标
    """
    print(f"\n{'='*60}")
    print(f"混淆类别深入分析")
    print(f"{'='*60}")

    n_classes = len(class_names)

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # 排除对角线，找 top_k 混淆对
    conf_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                conf_pairs.append((i, j, cm_norm[i, j], cm[i, j]))
    conf_pairs.sort(key=lambda x: x[3], reverse=True)  # 按绝对混淆数量排序

    top_pairs = conf_pairs[:top_k * 2]
    seen_pairs = set()
    unique_pairs = []
    for a, b, r, c in top_pairs:
        pair = tuple(sorted([a, b]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_pairs.append((a, b, r, c))
        if len(unique_pairs) >= top_k:
            break

    print(f"\n  Top {len(unique_pairs)} 最混淆类别对:")
    for a, b, r, c in unique_pairs:
        name_a = class_names[a] if a < len(class_names) else f'cls_{a}'
        name_b = class_names[b] if b < len(class_names) else f'cls_{b}'
        print(f"    {name_a:<25s} ↔ {name_b:<25s}  "
              f"conf_rate={r:.3f}  n={c}")

    # 在 PCA 空间中可视化
    from sklearn.preprocessing import StandardScaler
    features_norm = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_norm)

    fig, axes = plt.subplots(1, min(len(unique_pairs), 3),
                             figsize=(6 * min(len(unique_pairs), 3), 5))
    if len(unique_pairs) == 1:
        axes = [axes]

    for idx, (a, b, r, c) in enumerate(unique_pairs[:3]):
        ax = axes[idx]
        name_a = class_names[a] if a < len(class_names) else f'cls_{a}'
        name_b = class_names[b] if b < len(class_names) else f'cls_{b}'

        mask_a = labels == a
        mask_b = labels == b

        ax.scatter(pca_result[mask_a, 0], pca_result[mask_a, 1],
                   s=15, alpha=0.5, label=f'{name_a}', color='steelblue')
        ax.scatter(pca_result[mask_b, 0], pca_result[mask_b, 1],
                   s=15, alpha=0.5, label=f'{name_b}', color='coral')

        # 标注误分类样本
        wrong = (labels == a) & (preds == b)
        if wrong.sum() > 0:
            ax.scatter(pca_result[wrong, 0], pca_result[wrong, 1],
                       s=30, alpha=0.8, edgecolors='red', facecolors='none',
                       linewidths=1.5, label=f'Misclassified (n={wrong.sum()})')

        ax.set_title(f'{name_a} ↔ {name_b}\n(conf_rate={r:.3f})', fontsize=9)
        ax.legend(fontsize=7)

    plt.tight_layout()
    sp = save_path or os.path.join(RESULT_DIR, 'confusable_classes.png')
    fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 混淆类别可视化已保存: {sp}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='IMU 模型可视化分析')
    parser.add_argument('--model', default='TCN',
                        choices=['TCN', 'AttnConvLSTM', 'CNNTransformerV2',
                                 'MultiStreamCNN', 'ResCNN1D', 'DeepConvLSTM'])
    parser.add_argument('--ckpt', default=None,
                        help='模型 checkpoint 路径')
    parser.add_argument('--features', default=None,
                        help='预提取的特征文件 (.npy)')
    parser.add_argument('--features_labels', default=None,
                        help='特征对应的标签文件 (.npy)')
    parser.add_argument('--domain_labels', default=None,
                        help='Domain 标签文件 (.npy)')
    parser.add_argument('--data_dir', default='data/processed/')
    parser.add_argument('--all', action='store_true', help='运行所有可视化')
    parser.add_argument('--tsne', action='store_true', help='t-SNE 可视化')
    parser.add_argument('--umap', action='store_true', help='UMAP 可视化')
    parser.add_argument('--attention', action='store_true', help='时序注意力可视化')
    parser.add_argument('--domain', action='store_true', help='Domain gap 分析')
    parser.add_argument('--confusable', action='store_true', help='混淆类别分析')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    do_all = args.all

    # ---- 加载元信息 ----
    meta_path = os.path.join(args.data_dir, 'dataset_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        class_names = meta['class_names']
    else:
        class_names = [f'class_{i}' for i in range(70)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 加载特征 ----
    features = None
    labels = None
    preds = None

    if args.features and os.path.exists(args.features):
        features = np.load(args.features)
        lbl_path = args.features_labels or args.features.replace('features_', 'features_').replace('.npy', '_labels.npy')
        if os.path.exists(lbl_path):
            labels = np.load(lbl_path)
        print(f"Loaded features: {features.shape}")

    if args.domain_labels and os.path.exists(args.domain_labels):
        domain_labels = np.load(args.domain_labels).astype(np.int64).flatten()
    else:
        domain_path = os.path.join(args.data_dir, 'domain_test.npy')
        if os.path.exists(domain_path):
            domain_labels = np.load(domain_path).astype(np.int64).flatten()
            # 截断到特征长度
            if features is not None:
                domain_labels = domain_labels[:len(features)]
        else:
            domain_labels = None

    # ---- 如果需要从模型提取特征 ----
    if features is None and args.ckpt and os.path.exists(args.ckpt):
        # 加载模型
        use_v3 = args.model in ['TCN', 'AttnConvLSTM', 'CNNTransformerV2', 'MultiStreamCNN']
        if use_v3:
            model = create_model_v3(args.model, num_classes=len(class_names),
                                    dropout=0.3).to(device)
        else:
            model = create_model_v2(args.model, num_classes=len(class_names),
                                    dropout=0.3).to(device)

        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # 加载测试数据并提取特征
        x_test = np.load(os.path.join(args.data_dir, 'x_test.npy')).astype(np.float32)
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy')).astype(np.int64).flatten()
        ds = GestureDataset.from_arrays(x_test, y_test, train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        all_features = []
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits, feats, _ = model(x, return_features=True)
                all_features.append(feats.cpu().numpy())
                all_labels.extend(y.numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels)
        preds = np.array(all_preds)
        print(f"Extracted features: {features.shape}")

    if features is None:
        print("Error: 需要提供 --features 或 --ckpt 来获取特征")
        print("Example: python src/visualization.py --model TCN --ckpt models/best_TCN_v3.pth --all")
        return

    # ---- 运行可视化 ----
    if do_all or args.tsne:
        visualize_tsne(features, labels, class_names)

    if do_all or args.umap:
        visualize_umap(features, labels, class_names)

    if (do_all or args.attention) and args.ckpt:
        visualize_temporal_attention(
            model, loader, device, class_names)

    if (do_all or args.domain) and domain_labels is not None:
        visualize_domain_gap(features, labels, domain_labels, class_names)

    if do_all or args.confusable:
        if preds is None:
            print("  [SKIP] 混淆分析需要预测标签 (--pred_labels)")
        else:
            analyze_confusable_classes(features, labels, preds, class_names)

    print(f"\nAll visualizations saved to {RESULT_DIR}")


if __name__ == '__main__':
    main()
