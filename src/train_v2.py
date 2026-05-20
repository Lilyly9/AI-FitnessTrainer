"""
增强版训练脚本 v2 — 多数据集 IMU 动作分类。

核心改进（修复版）:
  1. WeightedRandomSampler — 类别平衡采样
  2. CrossEntropyLoss + class_weights + label_smoothing（稳定）
  3. 可选 FocalLoss（gamma 可调）
  4. 线性 warmup + CosineAnnealingWarmRestarts
  5. Gradient Clipping
  6. 早停 (macro F1)
  7. 完整评估: 逐类F1、混淆矩阵

用法:
  python -X utf8 src/train_v2.py                           # 默认 ResCNN1D, 100 epochs
  python -X utf8 src/train_v2.py --model DeepConvLSTM      # 切换架构
  python -X utf8 src/train_v2.py --loss focal --gamma 1.0   # FocalLoss
  python -X utf8 src/train_v2.py --epochs 200 --warmup 10   # 自定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os, sys, json, time, argparse
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v2 import create_model_v2
from dataset import GestureDataset

DATA_DIR = 'data/processed/'
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ============================================================
# FocalLoss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()


# ============================================================
# Train / Validate
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    p = np.array(all_preds); l = np.array(all_labels)
    return (total_loss / len(l), np.mean(p == l),
            f1_score(l, p, average='macro', zero_division=0), p, l)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='HAR Training v2')
    parser.add_argument('--model', default='ResCNN1D',
                        choices=['ResCNN1D', 'DeepConvLSTM', 'Gesture1DCNN',
                                 'TCN', 'AttnConvLSTM', 'CNNTransformerV2',
                                 'MultiStreamCNN'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--t0', type=int, default=30)
    parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt', default='models/best_model_v2.pth')
    parser.add_argument('--save_path', default='models/best_model_v2.pth')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Seed: {args.seed}")

    # ---- Meta ----
    with open(os.path.join(DATA_DIR, 'dataset_meta.json')) as f:
        meta = json.load(f)
    n_cls = meta['num_classes']; cls_names = meta['class_names']
    print(f"Classes: {n_cls}  Model: {args.model}  Loss: {args.loss}")

    # ---- Data ----
    x_train = np.load(DATA_DIR + 'x_train.npy').astype(np.float32)
    y_train = np.load(DATA_DIR + 'y_train.npy').astype(np.int64).flatten()
    x_test = np.load(DATA_DIR + 'x_test.npy').astype(np.float32)
    y_test = np.load(DATA_DIR + 'y_test.npy').astype(np.int64).flatten()
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    # ---- Stratified train/val split ----
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.2,
        random_state=args.seed, stratify=y_train)

    tr_ds = GestureDataset.from_arrays(x_train[tr_idx], y_train[tr_idx], train=True)
    va_ds = GestureDataset.from_arrays(x_train[va_idx], y_train[va_idx], train=False)
    te_ds = GestureDataset.from_arrays(x_test, y_test, train=False)

    # ---- Class weights & WeightedRandomSampler ----
    cnt = Counter(y_train[tr_idx])
    cls_w = torch.zeros(n_cls)
    for c in range(n_cls):
        cls_w[c] = len(tr_idx) / (n_cls * max(cnt.get(c, 1), 1))
    cls_w = cls_w.clamp(0.1, 10.0)

    sw = np.array([cls_w[l] for l in y_train[tr_idx]])
    sampler = WeightedRandomSampler(
        torch.tensor(sw, dtype=torch.float64), len(tr_idx), replacement=True)

    tr_ldr = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    va_ldr = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_ldr = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Model ----
    model = create_model_v2(args.model, num_classes=n_cls, dropout=args.dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- Loss ----
    if args.loss == 'focal':
        criterion = FocalLoss(weight=cls_w.to(device), gamma=args.gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_w.to(device),
                                        label_smoothing=args.label_smoothing)

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # ---- Scheduler: warmup then cosine ----
    cos = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=2, eta_min=1e-5)

    def warmup_step(epoch):
        if epoch < args.warmup:
            lr = args.lr * (epoch + 1) / args.warmup
            for g in optimizer.param_groups:
                g['lr'] = lr
        else:
            cos.step(epoch - args.warmup)

    # ---- Evaluate only ----
    if args.eval:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        _, acc, mf1, preds, labels = validate(model, te_ldr, criterion, device)
        print(f"Test Acc: {acc:.4f}  Macro F1: {mf1:.4f}")
        return

    # ---- Training ----
    best_mf1, best_ep, wait = 0.0, 0, 0
    hist = {'loss': [], 'val_acc': [], 'val_f1': []}
    t0 = time.time()

    for ep in range(args.epochs):
        warmup_step(ep)

        loss, tacc = train_epoch(model, tr_ldr, criterion, optimizer, device, args.grad_clip)
        vloss, vacc, vmf1, vp, vl = validate(model, va_ldr, criterion, device)

        hist['loss'].append(loss)
        hist['val_acc'].append(vacc)
        hist['val_f1'].append(vmf1)

        lr = optimizer.param_groups[0]['lr']
        mark = ''
        if vmf1 > best_mf1:
            best_mf1, best_ep, wait = vmf1, ep, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'best_macro_f1': best_mf1, 'epoch': ep,
                        'model_name': args.model, 'num_classes': n_cls,
                        'class_names': cls_names},
                       args.save_path)
            mark = ' *'
        else:
            wait += 1

        print(f"Ep {ep+1:3d} | loss={loss:.4f} | vacc={vacc:.4f} "
              f"| mf1={vmf1:.4f} | lr={lr:.6f} | {time.time()-t0:.0f}s{mark}")
        t0 = time.time()

        if wait >= args.patience:
            print(f"Early stop at epoch {ep+1} (best mf1={best_mf1:.4f} @ ep {best_ep+1})")
            break

    # ---- Load best & test ----
    ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    _, tacc, tmf1, tpreds, tlabels = validate(model, te_ldr, criterion, device)

    print(f"\n{'='*60}")
    print(f"FINAL: Acc={tacc:.4f} ({tacc*n_cls:.1f}x baseline)  Macro F1={tmf1:.4f}")
    print(f"{'='*60}")

    # ---- Classification report ----
    report = classification_report(tlabels, tpreds, labels=list(range(n_cls)),
                                   target_names=cls_names, zero_division=0, digits=3)
    with open('results/classification_report_v2.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Show per-class F1
    rdict = classification_report(tlabels, tpreds, labels=list(range(n_cls)),
                                  target_names=cls_names, output_dict=True, zero_division=0)
    pcls = [(n, rdict[n]['f1-score'], rdict[n]['support'])
            for n in cls_names if n in rdict]
    pcls.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 15 by F1:")
    for n, f1, s in pcls[:15]:
        print(f"  f1={f1:.3f}  {n:<45s} n={s:.0f}")
    nz = sum(1 for _, f1, _ in pcls if f1 == 0)
    print(f"  ...  F1=0: {nz}/{n_cls}")

    # ---- Confusion matrix ----
    cm = confusion_matrix(tlabels, tpreds, labels=list(range(n_cls)))
    cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(30, 28))
    ax.imshow(cm_n, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(cls_names, rotation=90, fontsize=5)
    ax.set_yticklabels(cls_names, fontsize=5)
    ax.set_title(f'{args.model} Confusion Matrix (Acc={tacc:.3f} MF1={tmf1:.3f})')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rng = range(1, len(hist['loss']) + 1)
    axes[0].plot(rng, hist['loss']); axes[0].set_title('Train Loss'); axes[0].grid(alpha=0.3)
    axes[1].plot(rng, hist['val_acc']); axes[1].set_title('Val Acc'); axes[1].grid(alpha=0.3)
    axes[2].plot(rng, hist['val_f1']); axes[2].set_title('Val Macro F1'); axes[2].grid(alpha=0.3)
    axes[2].axhline(y=best_mf1, color='r', linestyle='--', alpha=0.5, label=f'Best={best_mf1:.3f}')
    axes[2].legend()
    fig.suptitle(f'{args.model} ({n_cls} classes)')
    plt.tight_layout()
    plt.savefig('results/training_curve_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Per-dataset breakdown ----
    if 'label_sets' in meta:
        print("\nPer-dataset:")
        for ds, lbls in meta['label_sets'].items():
            m = np.isin(tlabels, lbls)
            if m.sum():
                a = np.mean(tpreds[m] == tlabels[m])
                f = f1_score(tlabels[m], tpreds[m], labels=lbls, average='macro', zero_division=0)
                print(f"  {ds}: acc={a:.4f}  mf1={f:.4f}  n={m.sum()}")

    print(f"\nModel saved: {args.save_path}")
    print("To demo: python -X utf8 src/demo.py")


if __name__ == '__main__':
    main()
