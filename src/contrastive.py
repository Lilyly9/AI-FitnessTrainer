"""
对比自监督预训练模块 (SimCLR-style for IMU time series)。

利用大量无标签 IMU 数据进行预训练，通过数据增强构造正样本对，
使用 NT-Xent 损失训练模型学习时序不变特征。

用法:
  python src/contrastive.py --pretrain --data data/processed/x_train.npy --epochs 100
  python src/contrastive.py --finetune --pretrained models/pretrain_simclr.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Gesture1DCNN
from dataset import GestureDataset


class ContrastiveAugment:
    """为对比学习生成两个不同的增强视图 (x_i, x_j) 作为正样本对。

    支持 5 种数据增强策略：
      - jitter:     添加高斯噪声
      - scaling:    随机幅度缩放
      - time_warp:  时间轴扭曲（插值重采样）
      - permutation: 时间片段随机排列
      - masking:    随机遮挡连续时间步
    """

    def __init__(self, noise_std=0.05, scale_range=(0.7, 1.3),
                 warp_range=(0.8, 1.2), shift_max=20, channel_drop_prob=0.2,
                 permute_segments=5, mask_ratio=0.1):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.warp_range = warp_range
        self.shift_max = shift_max
        self.channel_drop_prob = channel_drop_prob
        self.permute_segments = permute_segments  # 排列增强的片段数
        self.mask_ratio = mask_ratio  # 遮挡比例 (0~1)

    def __call__(self, x):
        x = x.copy()
        # 1) Jitter: 高斯噪声
        if np.random.rand() < 0.8:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
        # 2) Scaling: 幅度缩放
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*self.scale_range, size=(x.shape[0], 1)).astype(np.float32)
            x = x * scale
        # 3) Time Warp: 时间轴扭曲
        if np.random.rand() < 0.5:
            t = np.arange(x.shape[1], dtype=np.float32)
            warp = np.random.uniform(*self.warp_range)
            for c in range(x.shape[0]):
                x[c] = np.interp(t, t * warp, x[c])
        # 4) Permutation: 时间片段随机排列
        if np.random.rand() < 0.4:
            x = self._permutation(x)
        # 5) Masking: 随机遮挡连续时间步
        if np.random.rand() < 0.4:
            x = self._masking(x)
        # 通道丢弃
        if np.random.rand() < self.channel_drop_prob:
            drop_ch = np.random.randint(0, x.shape[0])
            x[drop_ch] = 0
        # 时间偏移
        if np.random.rand() < 0.3:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            if shift != 0:
                x = np.roll(x, shift, axis=1)
                if shift > 0:
                    x[:, :shift] = 0
                else:
                    x[:, shift:] = 0
        return x

    def _permutation(self, x):
        """将时间轴切分为若干段，随机打乱顺序后拼接。
        增强模型对局部时序顺序变化的不变性。
        """
        T = x.shape[1]
        n_seg = self.permute_segments
        # 随机化片段数 (3 ~ permute_segments)
        n_seg = np.random.randint(3, max(4, n_seg + 1))
        seg_len = T // n_seg
        if seg_len < 4:
            return x  # 窗口太短，不切分
        # 切分
        segments = []
        for i in range(n_seg):
            start = i * seg_len
            end = T if i == n_seg - 1 else (i + 1) * seg_len
            segments.append(x[:, start:end])
        # 随机打乱
        indices = np.arange(len(segments))
        np.random.shuffle(indices)
        return np.concatenate([segments[i] for i in indices], axis=1)

    def _masking(self, x):
        """随机遮挡一段连续时间步（类似时间域的 Cutout）。
        强制模型利用全局上下文而非局部模式。
        """
        T = x.shape[1]
        mask_len = max(4, int(T * np.random.uniform(0.05, self.mask_ratio)))
        if mask_len >= T - 1:
            return x
        start = np.random.randint(0, T - mask_len)
        x[:, start:start + mask_len] = 0.0
        return x


class ContrastiveDataset(Dataset):
    """对比学习数据集：每个样本生成两个增强视图。"""

    def __init__(self, data, augment=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.augment = augment or ContrastiveAugment()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.tensor(self.augment(x)), torch.tensor(self.augment(x))


class ProjectionHead(nn.Module):
    """MLP projection head：将 backbone 特征映射到对比学习的嵌入空间。"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SimCLRModel(nn.Module):
    """SimCLR 包装：backbone + projection head。"""

    def __init__(self, backbone, feature_dim, proj_dim=64):
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionHead(feature_dim, hidden_dim=128, output_dim=proj_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        return embeddings


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss。"""
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).view(2 * batch_size, 1)
    mask = torch.ones((2 * batch_size, 2 * batch_size), device=z.device).fill_diagonal_(0)
    mask[batch_size:, :batch_size] -= torch.eye(batch_size, device=z.device)
    mask[:batch_size, batch_size:] -= torch.eye(batch_size, device=z.device)
    negative_samples = sim[mask.bool()].view(2 * batch_size, -1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    return F.cross_entropy(logits, labels)


def pretrain_simclr(data, backbone_name='Gesture1DCNN', batch_size=64,
                    epochs=100, lr=1e-3, temperature=0.5, proj_dim=64,
                    device=None, save_path='models/pretrained_encoder.pth'):
    """SimCLR 自监督预训练。"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.asarray(data, dtype=np.float32)
    input_channels = data.shape[1]

    # 估计特征维度
    temp_model = Gesture1DCNN(input_channels=input_channels, num_classes=1)
    temp_model.fc = nn.Identity()
    with torch.no_grad():
        feat_dim = temp_model(torch.randn(1, input_channels, data.shape[2])).shape[1]
    del temp_model

    print(f"数据形状: {data.shape}, 特征维度: {feat_dim}")

    backbone = Gesture1DCNN(input_channels=input_channels, num_classes=1)
    backbone.fc = nn.Identity()
    model = SimCLRModel(backbone, feat_dim, proj_dim).to(device)

    dataset = ContrastiveDataset(data, ContrastiveAugment())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xi, xj in loader:
            xi, xj = xi.to(device), xj.to(device)
            optimizer.zero_grad()
            loss = nt_xent_loss(model(xi), model(xj), temperature)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xi.size(0)
        scheduler.step()
        avg_loss = epoch_loss / len(loader.dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(backbone.state_dict(), save_path)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

    print(f"预训练完成, best_loss={best_loss:.4f}, 保存到 {save_path}")
    return save_path


def finetune_linear_eval(pretrained_path, x_train, y_train, x_test, y_test,
                          num_classes=5, backbone_name='Gesture1DCNN',
                          batch_size=32, epochs=50, lr=1e-2, device=None):
    """线性探测评估。"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_channels = x_train.shape[1]

    backbone = Gesture1DCNN(input_channels=input_channels, num_classes=1)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
    backbone.to(device)
    backbone.eval()

    with torch.no_grad():
        feat_dim = backbone(torch.randn(1, input_channels, x_train.shape[2]).to(device)).shape[1]

    classifier = nn.Linear(feat_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = GestureDataset.from_arrays(x_train, y_train, train=False)
    test_ds = GestureDataset.from_arrays(x_test, y_test, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        classifier.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                features = backbone(inputs)
            loss = criterion(classifier(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(backbone(inputs))
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}: test_acc={acc:.4f}")

    print(f"线性探测最佳准确率: {best_acc:.4f}")
    return {'test_acc': best_acc}


def main():
    parser = argparse.ArgumentParser(description='对比学习自监督预训练')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--linear_eval', action='store_true')
    parser.add_argument('--data', default='data/processed/x_train.npy')
    parser.add_argument('--pretrained', default='models/pretrained_encoder.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/processed/'

    if args.pretrain:
        if not os.path.exists(args.data):
            print(f"数据 {args.data} 不存在")
            sys.exit(1)
        data = np.load(args.data).astype(np.float32)
        pretrain_simclr(data, epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, temperature=args.temperature, device=device,
                        save_path=args.pretrained)

    if args.finetune or args.linear_eval:
        if not os.path.exists(args.pretrained):
            print(f"预训练权重 {args.pretrained} 不存在，请先 --pretrain")
            sys.exit(1)
        paths = [os.path.join(data_dir, f) for f in ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']]
        if not all(os.path.exists(p) for p in paths):
            print("缺少预处理文件")
            sys.exit(1)
        x_tr = np.load(paths[0]).astype(np.float32)
        y_tr = np.load(paths[1]).astype(np.int64).flatten()
        x_te = np.load(paths[2]).astype(np.float32)
        y_te = np.load(paths[3]).astype(np.int64).flatten()
        finetune_linear_eval(args.pretrained, x_tr, y_tr, x_te, y_te,
                             epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, device=device)


if __name__ == '__main__':
    main()
