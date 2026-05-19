"""
多模型集成训练脚本（3 模型 Ensemble）。

自动使用 3 个不同随机种子训练 Gesture1DCNN，
推理时使用 soft voting（平均 logits）和平均概率融合。

用法:
  python src/train_ensemble.py                      # 默认训练 3 个模型
  python src/train_ensemble.py --evaluate            # 仅评估已训练模型
  python src/train_ensemble.py --seeds 42 3407 2025  # 自定义种子
  python src/train_ensemble.py --resume              # 从已有 checkpoint 恢复训练

输出:
  models/ensemble/model_1.pth   # seed=42
  models/ensemble/model_2.pth   # seed=3407
  models/ensemble/model_3.pth   # seed=2025
  models/ensemble/best_ensemble.pth  # 验证集最佳单模型副本
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import time
import argparse
from collections import Counter
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Gesture1DCNN, create_model
from dataset import GestureDataset

# ── 路径常量 ──
DATA_DIR = 'data/processed/'
ENSEMBLE_DIR = 'models/ensemble/'

# ── 默认超参数 ──
BATCH_SIZE = 64
EPOCHS = 80
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
VAL_SPLIT = 0.2
MAX_PATIENCE = 20

# ── 默认 3 个随机种子 ──
DEFAULT_SEEDS = [42, 3407, 2025]


def set_seed(seed):
    """固定随机种子以保证可复现性。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y_path=os.path.join(DATA_DIR, 'y_train.npy')):
    """反频率类别权重——样本少的类权重大。"""
    y = np.load(y_path)
    counter = Counter(y)
    total = len(y)
    n_classes = len(counter)
    weights = np.ones(n_classes, dtype=np.float32)
    for cls, count in counter.items():
        weights[cls] = total / (n_classes * count)
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss — 自动聚焦困难样本，天然处理类别不均衡。

    FL = -(1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        logp = torch.log_softmax(pred, dim=1)
        pt = torch.exp(logp)

        if self.label_smoothing > 0:
            n_classes = pred.size(1)
            smooth_target = torch.full_like(pt, self.label_smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(smooth_target * logp).sum(dim=1)
        else:
            ce = torch.nn.functional.nll_loss(logp, target, reduction='none')

        p_t = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce

        if self.weight is not None:
            loss = loss * self.weight[target]

        return loss.mean()


def train_one_model(seed, num_classes, class_weights, device, model_idx,
                    batch_size=64, epochs=80, lr=0.01,
                    resume_path=None, pretrained_path=None):
    """训练单个模型。

    参数:
      seed: 随机种子
      num_classes: 类别数
      class_weights: 类别权重 tensor
      device: 训练设备
      model_idx: 模型编号 (1-based, 用于保存命名)
      batch_size: 批次大小
      epochs: 训练轮数
      lr: 学习率
      resume_path: 恢复训练的 checkpoint 路径（可选）

    返回:
      model_path: 最佳模型保存路径
      best_acc: 最佳验证准确率
      history: {'train_loss': [], 'val_acc': []}
    """
    set_seed(seed)

    # ── 数据加载 ──
    ds_aug = GestureDataset(
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'), train=True)
    ds_noaug = GestureDataset(
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'), train=False)

    gen = torch.Generator().manual_seed(seed)
    num_val = int(len(ds_aug) * VAL_SPLIT)
    num_train = len(ds_aug) - num_val
    train_ds, _ = random_split(ds_aug, [num_train, num_val], generator=gen)
    _, val_ds = random_split(ds_noaug, [num_train, num_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # ── 模型 ──
    model = Gesture1DCNN(input_channels=6, num_classes=num_classes,
                         dropout=0.5).to(device)

    # 加载预训练 encoder 权重（strict=False 跳过 fc 层）
    if pretrained_path and os.path.exists(pretrained_path):
        pretrained_state = torch.load(pretrained_path, map_location=device)
        if isinstance(pretrained_state, dict) and 'model_state_dict' in pretrained_state:
            pretrained_state = pretrained_state['model_state_dict']
        model.load_state_dict(pretrained_state, strict=False)
        print(f'  已加载预训练权重: {pretrained_path}')

    # 如果提供了 resume 路径，加载已有权重继续训练
    start_epoch = 1
    best_acc = 0.0
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 1) + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f'  从 {resume_path} 恢复训练 (epoch={start_epoch}, best_acc={best_acc:.4f})')
        else:
            model.load_state_dict(checkpoint)
            print(f'  从 {resume_path} 加载权重')

    # ── 损失函数与优化器 ──
    weights_gpu = class_weights.to(device)
    criterion = FocalLoss(weight=weights_gpu, gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-5)

    # 如果恢复训练，也需要恢复优化器和调度器状态
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    model_path = os.path.join(ENSEMBLE_DIR, f'model_{model_idx}.pth')
    history = {'train_loss': [], 'val_acc': []}
    patience_counter = 0

    # ── 训练循环 ──
    for epoch in range(start_epoch, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # Val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total

        avg_loss = train_loss / num_train
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型（含完整训练状态，便于 resume）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'  [Model {model_idx}] Epoch {epoch:3d} | loss={avg_loss:.4f} | '
                  f'val_acc={val_acc:.4f} | best={best_acc:.4f}')

        if patience_counter >= MAX_PATIENCE:
            print(f'  [Model {model_idx}] 早停于 Epoch {epoch} (最佳={best_acc:.4f})')
            break

    # 保存最终模型权重（纯 state_dict，方便 demo.py 加载）
    final_state = torch.load(model_path, map_location=device)
    torch.save(final_state['model_state_dict'],
               os.path.join(ENSEMBLE_DIR, f'model_{model_idx}_weights.pth'))

    return model_path, best_acc, history


def ensemble_predict(models, device, loader):
    """Soft voting 集成预测：平均 logits 后取 argmax。

    返回:
      all_preds: 预测标签
      all_labels: 真实标签
      all_probs: 平均概率 (N, num_classes)
      per_model_probs: 各模型概率 [(N, num_classes), ...]
    """
    for m in models:
        m.eval()

    all_preds, all_labels = [], []
    all_probs_list = []
    per_model_logits = [[] for _ in models]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 收集各模型 logits
            batch_logits = []
            for i, m in enumerate(models):
                logits = m(inputs)
                batch_logits.append(logits)
                per_model_logits[i].append(logits.cpu().numpy())

            # Soft voting: 平均 logits → softmax → argmax
            avg_logits = torch.stack(batch_logits).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=1)
            preds = torch.argmax(avg_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_list.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs_list, axis=0)

    # 各模型概率
    per_model_probs = []
    for logit_list in per_model_logits:
        logits_np = np.concatenate(logit_list, axis=0)
        per_model_probs.append(torch.softmax(torch.tensor(logits_np), dim=1).numpy())

    return all_preds, all_labels, all_probs, per_model_probs


def print_per_class_accuracy(preds, labels, class_names, num_classes):
    """输出每类准确率。"""
    print('\n' + '=' * 70)
    print('每类准确率 (Per-Class Accuracy)')
    print('=' * 70)
    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        if mask.sum() > 0:
            acc = (preds[mask] == cls_idx).mean()
            bar = '█' * int(acc * 40) + '░' * (40 - int(acc * 40))
            print(f'  {class_names[cls_idx]:<30s} | {bar} | {acc:.4f}  (n={mask.sum()})')
        else:
            print(f'  {class_names[cls_idx]:<30s} | {"(无样本)":>42s}')


def main():
    parser = argparse.ArgumentParser(description='多模型集成训练')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'训练种子列表 (默认: {DEFAULT_SEEDS})')
    parser.add_argument('--evaluate', action='store_true',
                        help='仅评估已训练模型，跳过训练')
    parser.add_argument('--resume', action='store_true',
                        help='从已有 checkpoint 恢复训练')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'训练轮数 (默认: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'批次大小 (默认: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'学习率 (默认: {LR})')
    parser.add_argument('--model_name', default='Gesture1DCNN',
                        choices=['Gesture1DCNN', 'GestureMultiScaleCNN'],
                        help='模型架构 (默认: Gesture1DCNN)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练 encoder 路径 (e.g. models/pretrained_encoder.pth)')
    args = parser.parse_args()

    seeds = args.seeds

    # ── 读取类别信息 ──
    meta_path = os.path.join(DATA_DIR, 'dataset_meta.json')
    if not os.path.exists(meta_path):
        print(f'错误: 找不到 {meta_path}，请先运行 merge_datasets.py')
        sys.exit(1)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    num_classes = meta['num_classes']
    class_names = meta['class_names']
    print(f'训练配置: {num_classes} 类, 数据集={meta.get("datasets", [])}')
    print(f'随机种子: {seeds}')
    print(f'模型架构: {args.model_name}')

    # ── 计算类别权重 ──
    y_all = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    counter = Counter(y_all)
    total = len(y_all)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, cnt in counter.items():
        weights[cls] = total / (num_classes * cnt)
    weights = np.clip(weights, 0.1, 10.0)
    class_weights = torch.tensor(weights)
    print(f'类别权重范围: [{weights.min():.2f}, {weights.max():.2f}]')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # ── 训练阶段 ──
    model_paths = []
    best_acc_list = []
    all_histories = []

    if not args.evaluate:
        t_start = time.time()
        for i, seed in enumerate(seeds):
            model_idx = i + 1
            print(f'\n{"=" * 50}')
            print(f'训练模型 {model_idx}/{len(seeds)} (seed={seed})')
            print(f'{"=" * 50}')

            resume_path = None
            if args.resume:
                resume_path = os.path.join(ENSEMBLE_DIR, f'model_{model_idx}.pth')
                if not os.path.exists(resume_path):
                    resume_path = None

            t0 = time.time()
            path, best_acc, hist = train_one_model(
                seed=seed, num_classes=num_classes,
                class_weights=class_weights, device=device,
                model_idx=model_idx, resume_path=resume_path,
                batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                pretrained_path=args.pretrained)
            model_paths.append(path)
            best_acc_list.append(best_acc)
            all_histories.append(hist)
            print(f'  [Model {model_idx}] 耗时: {time.time() - t0:.1f}s, '
                  f'最佳 val_acc={best_acc:.4f}')

        t_train = time.time() - t_start
        print(f'\n训练总耗时: {t_train:.1f}s ({t_train/60:.1f}min)')
        print(f'各模型最佳 val_acc: {[f"{a:.4f}" for a in best_acc_list]}')
        print(f'平均 val_acc: {np.mean(best_acc_list):.4f} ± {np.std(best_acc_list):.4f}')
    else:
        # 仅评估模式：直接从已有文件加载路径
        for i in range(len(seeds)):
            p = os.path.join(ENSEMBLE_DIR, f'model_{i+1}.pth')
            if os.path.exists(p):
                model_paths.append(p)
            else:
                # 尝试 weights 版本
                wp = os.path.join(ENSEMBLE_DIR, f'model_{i+1}_weights.pth')
                if os.path.exists(wp):
                    model_paths.append(wp)
                else:
                    print(f'警告: {p} 不存在，跳过模型 {i+1}')
        if not model_paths:
            print('错误: 未找到任何已训练模型，请先运行训练。')
            sys.exit(1)
        print(f'找到 {len(model_paths)} 个已训练模型')

    # ── 加载集成模型 ──
    ensemble_models = []
    for path in model_paths:
        m = create_model(model_name=args.model_name, input_channels=6,
                         num_classes=num_classes, dropout=0.5).to(device)
        state = torch.load(path, map_location=device)
        # 兼容完整 checkpoint 和纯 state_dict
        if isinstance(state, dict) and 'model_state_dict' in state:
            m.load_state_dict(state['model_state_dict'])
        else:
            m.load_state_dict(state)
        m.eval()
        ensemble_models.append(m)
    print(f'已加载 {len(ensemble_models)} 个模型')

    # ── 测试集评估 ──
    t_eval_start = time.time()
    test_ds = GestureDataset(
        os.path.join(DATA_DIR, 'x_test.npy'),
        os.path.join(DATA_DIR, 'y_test.npy'), train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    all_preds, all_labels, all_probs, per_model_probs = ensemble_predict(
        ensemble_models, device, test_loader)

    ensemble_acc = np.mean(all_preds == all_labels)
    print(f'\n{"=" * 70}')
    print(f'集成模型测试集准确率 (Soft Voting): {ensemble_acc:.4f}')
    print(f'随机基线 ({num_classes}类): {1.0/num_classes:.4f}')
    print(f'{"=" * 70}')

    # ── 各模型单独评估 ──
    print('\n单模型 vs 集成对比:')
    print('-' * 50)
    for i, (path, m) in enumerate(zip(model_paths, ensemble_models)):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = m(inputs).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        single_acc = correct / total
        print(f'  Model {i+1} (seed={seeds[i]}): {single_acc:.4f}')
    print(f'  Ensemble (Soft Voting):     {ensemble_acc:.4f}  ←')
    print(f'  集成提升:                   {ensemble_acc - np.mean([(correct/total) for m in ensemble_models]):+.4f}')

    # ── 平均概率融合（对比 soft voting） ──
    avg_probs_ensemble = np.mean(per_model_probs, axis=0)
    prob_fusion_preds = np.argmax(avg_probs_ensemble, axis=1)
    prob_fusion_acc = np.mean(prob_fusion_preds == all_labels)
    print(f'\n  平均概率融合准确率:           {prob_fusion_acc:.4f}')

    # ── 每类平均概率 ──
    print('\n' + '=' * 70)
    print('每类平均预测概率 (Top-5 展示)')
    print('=' * 70)
    per_class_mean_prob = []
    for cls_idx in range(num_classes):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            cls_mean_prob = all_probs[mask].mean(axis=0)
            per_class_mean_prob.append((cls_idx, cls_mean_prob))
    # 按对角线概率排序（模型对各类的置信度）
    per_class_mean_prob.sort(key=lambda x: x[1][x[0]], reverse=True)
    for rank, (cls_idx, probs) in enumerate(per_class_mean_prob[:5]):
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_str = ' | '.join(f'{class_names[i]}: {probs[i]:.3f}'
                              for i in top3_indices)
        print(f'  {rank+1}. {class_names[cls_idx]:<30s} → {top3_str}')

    # ── 每类准确率 ──
    print_per_class_accuracy(all_preds, all_labels, class_names, num_classes)

    # ── 保存评估结果 ──
    os.makedirs('results', exist_ok=True)
    with open('results/ensemble_eval.txt', 'w', encoding='utf-8') as f:
        f.write(f'集成模型测试准确率: {ensemble_acc:.4f}\n')
        f.write(f'平均概率融合准确率: {prob_fusion_acc:.4f}\n')
        f.write(f'随机基线: {1.0/num_classes:.4f}\n\n')
        f.write(f'各模型准确率:\n')
        for i, (path, m) in enumerate(zip(model_paths, ensemble_models)):
            m.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = m(inputs).argmax(1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
            f.write(f'  Model {i+1}: {correct/total:.4f}\n')
        f.write(f'\n每类准确率:\n')
        for cls_idx in range(num_classes):
            mask = all_labels == cls_idx
            if mask.sum() > 0:
                acc = (all_preds[mask] == cls_idx).mean()
                f.write(f'  {class_names[cls_idx]}: {acc:.4f} (n={mask.sum()})\n')

    print(f'\n评估结果已保存到 results/ensemble_eval.txt')
    print(f'总耗时: {time.time() - t_eval_start + (0 if args.evaluate else time.time() - t_eval_start):.1f}s')


if __name__ == '__main__':
    main()
