"""
多类分类模型训练（单模型，CPU/GPU 通用，类别数自动从 dataset_meta.json 读取）。
训练完成后权重保存到 models/best_model.pth。

用法:
  python src/train_67class.py                                  # 默认训练
  python src/train_67class.py --pretrained models/pretrained_encoder.pth  # 预训练微调
  python src/train_67class.py --resume models/best_model.pth   # 恢复训练
"""

import torch, torch.nn as nn, numpy as np, os, sys, json, time, argparse
from collections import Counter
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Gesture1DCNN
from dataset import GestureDataset

DATA_DIR = 'data/processed/'


class FocalLoss(nn.Module):
    """Focal Loss — 自动聚焦困难样本，天然处理类别不均衡。"""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        logp = torch.log_softmax(pred, dim=1)
        pt = torch.exp(logp)
        if self.label_smoothing > 0:
            n = pred.size(1)
            sm = torch.full_like(pt, self.label_smoothing / (n - 1))
            sm.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(sm * logp).sum(dim=1)
        else:
            ce = torch.nn.functional.nll_loss(logp, target, reduction='none')
        p_t = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = (1 - p_t) ** self.gamma * ce
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def main():
    parser = argparse.ArgumentParser(description='多类分类模型训练（类别数自动从 dataset_meta.json 读取）')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练 encoder 路径 (e.g. models/pretrained_encoder.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--epochs', type=int, default=80,
                        help='训练轮数 (默认: 80)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率 (默认: 0.01)')
    parser.add_argument('--save_path', type=str, default='models/best_model.pth',
                        help='模型保存路径')
    args = parser.parse_args()

    # ── 读取类别信息 ──
    with open(os.path.join(DATA_DIR, 'dataset_meta.json')) as f:
        meta = json.load(f)
    num_classes = meta['num_classes']
    class_names = meta['class_names']
    print(f'训练配置: {num_classes} 类, {len(class_names)} 个类名')

    # ── 动态权重 ──
    y_all = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    counter = Counter(y_all)
    total = len(y_all)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, cnt in counter.items():
        weights[cls] = total / (num_classes * cnt)
    weights = np.clip(weights, 0.1, 10.0)
    class_weights = torch.tensor(weights)
    print(f'类别权重范围: [{weights.min():.2f}, {weights.max():.2f}]')

    # ── 加载数据 ──
    ds_aug = GestureDataset(DATA_DIR + 'x_train.npy', DATA_DIR + 'y_train.npy', train=True)
    ds_noaug = GestureDataset(DATA_DIR + 'x_train.npy', DATA_DIR + 'y_train.npy', train=False)
    gen = torch.Generator().manual_seed(42)
    nv = int(len(ds_aug) * 0.2)
    nt = len(ds_aug) - nv
    train_ds, _ = random_split(ds_aug, [nt, nv], generator=gen)
    _, val_ds = random_split(ds_noaug, [nt, nv], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'训练窗口: {nt}, 验证窗口: {nv}')

    # ── 模型 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    model = Gesture1DCNN(input_channels=6, num_classes=num_classes, dropout=0.5).to(device)

    # 加载预训练 encoder 权重（strict=False 跳过形状不匹配的层）
    if args.pretrained and os.path.exists(args.pretrained):
        pretrained_state = torch.load(args.pretrained, map_location=device)
        if isinstance(pretrained_state, dict) and 'model_state_dict' in pretrained_state:
            pretrained_state = pretrained_state['model_state_dict']
        model.load_state_dict(pretrained_state, strict=False)
        print(f'已加载预训练权重: {args.pretrained}')

    # 恢复训练
    start_epoch = 1
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 1) + 1
            best_acc = checkpoint.get('best_acc', 0.0)
        else:
            model.load_state_dict(checkpoint)
        print(f'从 {args.resume} 恢复训练 (start_epoch={start_epoch})')

    criterion = FocalLoss(weight=class_weights.to(device), gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-5)

    # 如果恢复训练，恢复优化器状态
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    os.makedirs(os.path.dirname(args.save_path) or 'models', exist_ok=True)
    patience_counter = 0
    MAX_PATIENCE = 20
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # ── Val ──
        model.eval()
        correct, total_v = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(1)
                total_v += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total_v

        elapsed = time.time() - t0
        print(f'Epoch {epoch:2d} | loss={train_loss/nt:.4f} | val_acc={val_acc:.4f} | {elapsed:.0f}s',
              flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, args.save_path)
            patience_counter = 0
            print(f'  => saved (best={best_acc:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= MAX_PATIENCE:
            print(f'早停于 epoch {epoch}')
            break

    total_time = time.time() - t_start
    print(f'\n训练完成: best_val_acc={best_acc:.4f}, 总时间={total_time/60:.1f}min')

    # ── 测试集评估 ──
    test_ds = GestureDataset(DATA_DIR + 'x_test.npy', DATA_DIR + 'y_test.npy', train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    checkpoint = torch.load(args.save_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    correct, total_t = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            total_t += labels.size(0)
            correct += (preds == labels).sum().item()
    test_acc = correct / total_t
    print(f'测试集准确率 (LOSO): {test_acc:.4f}')
    print(f'权重已保存到 {args.save_path}')


if __name__ == '__main__':
    main()
