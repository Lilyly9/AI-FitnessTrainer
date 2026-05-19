"""
CNN + Transformer 混合模型训练脚本。

使用 GestureCNNTransformer 进行多类 IMU 动作分类（类别数自动从 dataset_meta.json 读取）。
CNN 提取局部时序特征 → Transformer 建模长程依赖 → 分类头输出。

用法:
  python src/train_transformer.py                          # 默认训练
  python src/train_transformer.py --epochs 120 --lr 0.001  # 自定义超参数
  python src/train_transformer.py --resume models/best_transformer.pth  # 恢复训练

模型结构:
  conv_stem (3层1D-CNN) → PositionalEncoding → TransformerEncoder → GlobalPool → FC

输出:
  models/best_transformer.pth   # 最佳模型 checkpoint
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
from model import GestureCNNTransformer, create_model
from dataset import GestureDataset

# ── 路径常量 ──
DATA_DIR = 'data/processed/'

# ── 默认超参数（适合 Transformer 的保守设置） ──
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001           # Transformer 通常用较低学习率
WEIGHT_DECAY = 1e-4  # Transformer 通常用较小 weight decay
VAL_SPLIT = 0.2
MAX_PATIENCE = 25
WARMUP_EPOCHS = 5    # 线性 warmup 轮数

# ── Transformer 结构参数 ──
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.5


def set_seed(seed=42):
    """固定随机种子。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_class_weights(y_path=os.path.join(DATA_DIR, 'y_train.npy')):
    """反频率类别权重——稀有类获得更高权重。"""
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
    """Focal Loss — 聚焦困难样本，天然处理类别不均衡。

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


def train(args):
    """主训练函数。

    参数全部从 argparse args 中读取，便于 CLI 和编程调用。
    """
    set_seed(42)

    # ── 加载元信息 ──
    meta_path = os.path.join(DATA_DIR, 'dataset_meta.json')
    if not os.path.exists(meta_path):
        print(f'错误: 找不到 {meta_path}，请先运行 merge_datasets.py')
        sys.exit(1)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    num_classes = meta['num_classes']
    class_names = meta['class_names']
    print(f'训练配置: {num_classes} 类')
    print(f'数据集: {meta.get("datasets", [])}')

    # ── 设备 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

    # ── 类别权重 ──
    y_all = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    counter = Counter(y_all)
    total_w = len(y_all)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, cnt in counter.items():
        weights[cls] = total_w / (num_classes * cnt)
    weights = np.clip(weights, 0.1, 10.0)
    class_weights = torch.tensor(weights).to(device)
    print(f'类别权重范围: [{weights.min():.2f}, {weights.max():.2f}]')

    # ── 数据加载 ──
    ds_aug = GestureDataset(
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'), train=True)
    ds_noaug = GestureDataset(
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'), train=False)

    gen = torch.Generator().manual_seed(42)
    num_val = int(len(ds_aug) * VAL_SPLIT)
    num_train = len(ds_aug) - num_val
    train_ds, _ = random_split(ds_aug, [num_train, num_val], generator=gen)
    _, val_ds = random_split(ds_noaug, [num_train, num_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    print(f'训练窗口: {num_train}, 验证窗口: {num_val}')

    # ── 模型 ──
    model = GestureCNNTransformer(
        input_channels=6, num_classes=num_classes, dropout=args.dropout,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward).to(device)

    # 加载预训练权重（如果架构兼容）
    if args.pretrained and os.path.exists(args.pretrained):
        pretrained_state = torch.load(args.pretrained, map_location=device)
        if isinstance(pretrained_state, dict) and 'model_state_dict' in pretrained_state:
            pretrained_state = pretrained_state['model_state_dict']
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        print(f'已加载预训练权重: {args.pretrained}')
        if missing:
            print(f'  未匹配层 (将随机初始化): {len(missing)} 层')
        if unexpected:
            print(f'  多余层 (已忽略): {len(unexpected)} 层')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数量: {num_params:,}')

    # ── 损失函数与优化器（AdamW 更适合 Transformer） ──
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # CosineAnnealingWarmRestarts：周期性重启，适合长训练
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # ── Resume 支持 ──
    start_epoch = 1
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_acc': []}

    os.makedirs('models', exist_ok=True)
    save_path = args.save_path or 'models/best_transformer.pth'

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        history = checkpoint.get('history', history)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f'从 Epoch {start_epoch} 恢复训练 (best_acc={best_acc:.4f})')

    # ── 训练循环 ──
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Transformer 梯度裁剪阈值略小
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # ---- Val ----
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

        elapsed = time.time() - t0

        # 每 5 轮或首尾打印
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | '
                  f'val_acc={val_acc:.4f} | lr={lr_now:.2e} | {elapsed:.1f}s',
                  flush=True)

        # ---- 保存最佳模型 ----
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'history': history,
                'patience_counter': patience_counter,
                'num_classes': num_classes,
                'class_names': class_names,
            }, save_path)
            print(f'  => 已保存最佳模型 (val_acc={best_acc:.4f})')
        else:
            patience_counter += 1

        # ---- Early Stopping ----
        if patience_counter >= args.patience:
            print(f'早停于 Epoch {epoch} (最佳 val_acc={best_acc:.4f})')
            break

    total_time = time.time() - t_start
    print(f'\n训练完成: best_val_acc={best_acc:.4f}, 总时间={total_time/60:.1f}min')

    # ── 测试集评估 ──
    print('\n' + '=' * 60)
    print('测试集评估')
    print('=' * 60)

    test_ds = GestureDataset(
        os.path.join(DATA_DIR, 'x_test.npy'),
        os.path.join(DATA_DIR, 'y_test.npy'), train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    # 加载最佳模型
    best_checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    correct, total_t = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            total_t += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct / total_t
    print(f'测试集准确率: {test_acc:.4f}')
    print(f'随机基线 ({num_classes}类): {1.0/num_classes:.4f}')

    # ── 每类准确率 ──
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    print(f'\n每类准确率:')
    for cls_idx in range(num_classes):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            acc = (all_preds[mask] == cls_idx).mean()
            bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
            print(f'  {class_names[cls_idx]:<30s} | {bar} | {acc:.4f} (n={mask.sum()})')

    # ── 保存评估报告 ──
    os.makedirs('results', exist_ok=True)
    with open('results/transformer_eval.txt', 'w', encoding='utf-8') as f:
        f.write(f'模型: GestureCNNTransformer\n')
        f.write(f'测试准确率: {test_acc:.4f}\n')
        f.write(f'最佳验证准确率: {best_acc:.4f}\n')
        f.write(f'参数量: {num_params:,}\n\n')
        f.write(f'每类准确率:\n')
        for cls_idx in range(num_classes):
            mask = all_labels == cls_idx
            if mask.sum() > 0:
                acc = (all_preds[mask] == cls_idx).mean()
                f.write(f'  {class_names[cls_idx]}: {acc:.4f} (n={mask.sum()})\n')
    print(f'\n评估报告已保存到 results/transformer_eval.txt')

    return model, best_acc, test_acc, history


def main():
    parser = argparse.ArgumentParser(
        description='CNN + Transformer 混合模型训练（类别数自动从 dataset_meta.json 读取）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'训练轮数 (默认: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'批次大小 (默认: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'学习率 (默认: {LR})')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help=f'权重衰减 (默认: {WEIGHT_DECAY})')
    parser.add_argument('--patience', type=int, default=MAX_PATIENCE,
                        help=f'早停耐心值 (默认: {MAX_PATIENCE})')

    # 模型结构参数
    parser.add_argument('--d_model', type=int, default=D_MODEL,
                        help=f'Transformer 隐层维度 (默认: {D_MODEL})')
    parser.add_argument('--nhead', type=int, default=NHEAD,
                        help=f'注意力头数 (默认: {NHEAD})')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS,
                        help=f'Transformer 层数 (默认: {NUM_LAYERS})')
    parser.add_argument('--dim_feedforward', type=int, default=DIM_FEEDFORWARD,
                        help=f'前馈网络维度 (默认: {DIM_FEEDFORWARD})')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help=f'Dropout 比例 (默认: {DROPOUT})')

    # 恢复与预训练
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练权重路径（需架构兼容）')
    parser.add_argument('--save_path', type=str, default=None,
                        help='模型保存路径 (默认: models/best_transformer.pth)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
