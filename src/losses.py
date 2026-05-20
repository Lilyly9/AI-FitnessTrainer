"""
损失函数模块 — 分类损失 + 度量学习损失。

包含:
  - FocalLoss: 聚焦困难样本，处理类别不平衡
  - LabelSmoothingCrossEntropy: 标签平滑，防止过拟合
  - SupervisedContrastiveLoss: 有监督对比损失，拉近同类/推远异类
  - TripletLoss: 三元组损失，直接优化嵌入空间
  - CombinedLoss: 组合损失，分类 + 对比的加权和

用于解决高相似动作 (lunge/pushup/situp/squat) 的区分问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# FocalLoss
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss — 自动降低简单样本权重，聚焦困难样本。

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    参数:
      gamma: 聚焦参数。gamma=0 → CE, gamma=2 → 强聚焦困难样本
      weight: 类别权重 (class_weights)
      label_smoothing: 标签平滑比例
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)

        if self.label_smoothing > 0:
            # 标签平滑
            smooth_target = torch.full_like(pred, self.label_smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            logp = F.log_softmax(pred, dim=1)
            ce = -(smooth_target * logp).sum(dim=1)
            p_t = torch.exp(logp).gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            logp = F.log_softmax(pred, dim=1)
            ce = F.nll_loss(logp, target, reduction='none')
            p_t = torch.exp(logp).gather(1, target.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma

        if self.weight is not None:
            focal_weight = focal_weight * self.weight[target]

        return (focal_weight * ce).mean()


# ============================================================
# LabelSmoothingCrossEntropy
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵 — 防止模型对标签过于自信。"""

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_labels * log_probs).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


# ============================================================
# SupervisedContrastiveLoss
# ============================================================
class SupervisedContrastiveLoss(nn.Module):
    """有监督对比损失 (Khosla et al., NeurIPS 2020)。

    在 batch 内：
      - 正样本对: 同类的所有其他样本
      - 负样本对: 不同类的所有样本
      - 拉近同类特征，推远异类特征

    用法:
      features = model.get_features(x)  # (B, D)
      loss = supcon(features, labels)

    注意:
      - 需要足够大的 batch_size (>= 128) 才有效
      - features 需要先 L2 归一化
      - 对高相似动作 (lunge/pushup/situp) 特别有效
    """

    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """features: (B, D), labels: (B,)"""
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 归一化
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature)

        # 排除自身 (对角线)
        logits_mask = torch.ones_like(anchor_dot_contrast).scatter_(
            1, torch.arange(batch_size).view(-1, 1).to(device), 0)

        # 找到正样本对 (同标签)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * logits_mask  # 排除自身

        # 计算 log-softmax
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # 正样本的平均 log 概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# ============================================================
# TripletLoss — 在线三元组损失
# ============================================================
class TripletLoss(nn.Module):
    """在线三元组损失 — 在 batch 内挖掘 hardest positive/negative。

    对于每个 anchor:
      - positive: 同类中最远的样本 (hardest positive)
      - negative: 异类中最近的样本 (hardest negative)
      - loss = max(0, d(a,p) - d(a,n) + margin)

    对高相似类别间区分特别有效。
    """

    def __init__(self, margin=0.3, mining='semi-hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining  # 'hard' | 'semi-hard' | 'all'

    def forward(self, features, labels):
        """features: (B, D), labels: (B,)"""
        device = features.device
        features = F.normalize(features, dim=1)

        # 成对距离矩阵
        dist_mat = 1.0 - torch.matmul(features, features.T)  # cosine distance
        dist_mat = dist_mat.clamp(min=0)

        batch_size = features.shape[0]
        labels = labels.view(-1, 1)

        # 正样本 mask (同类，排除自身)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        # 负样本 mask (异类)
        neg_mask = 1.0 - torch.eq(labels, labels.T).float().to(device)

        if self.mining == 'hard':
            # Hardest positive: 同类中最远的
            pos_dist = (dist_mat * pos_mask).max(dim=1)[0]

            # Hardest negative: 异类中最近的
            neg_dist = (dist_mat + 1e6 * (1 - neg_mask)).min(dim=1)[0]

        elif self.mining == 'semi-hard':
            # Semi-hard: negative > positive 且在 margin 内
            pos_dist = (dist_mat * pos_mask).max(dim=1)[0]

            neg_dist_semi = dist_mat.clone()
            # 不符合 semi-hard 条件的设为 -inf
            valid = (neg_dist_semi > pos_dist.unsqueeze(1)) & neg_mask.bool()
            neg_dist_semi[~valid] = -float('inf')

            # 取最近的 semi-hard negative
            neg_dist_hard = (neg_dist_semi + 1e-6).max(dim=1)[0]

            # 没有 semi-hard negative 的样本用 hardest negative
            no_valid = (neg_dist_hard <= 0)
            if no_valid.any():
                neg_dist_hard[no_valid] = (
                    (dist_mat[no_valid] + 1e6 * (1 - neg_mask[no_valid])).min(dim=1)[0]
                )
            neg_dist = neg_dist_hard

        else:  # 'all'
            # 使用所有正负对
            pos_dist = (dist_mat * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-10)
            neg_dist = (dist_mat * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-10)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ============================================================
# CombinedLoss — 组合损失
# ============================================================
class CombinedLoss(nn.Module):
    """分类损失 + 对比损失 + 三元组损失的加权组合。

    total = λ_ce * CE + λ_supcon * SupCon + λ_trip * Triplet

    推荐调度:
      初期: λ_supcon=0, λ_trip=0 → 纯分类快速收敛
      中期: λ_supcon=0.1, λ_trip=0.05 → 开始优化嵌入
      后期: λ_supcon=0.2, λ_trip=0.1 → 强化类间分离
    """

    def __init__(self, num_classes=70, class_weights=None,
                 ce_weight=1.0, supcon_weight=0.1, triplet_weight=0.05,
                 focal_gamma=2.0, label_smoothing=0.05,
                 supcon_temperature=0.1, triplet_margin=0.3):
        super().__init__()

        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.triplet_weight = triplet_weight

        self.ce_loss = FocalLoss(
            weight=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        self.supcon_loss = SupervisedContrastiveLoss(
            temperature=supcon_temperature,
        )
        self.triplet_loss = TripletLoss(
            margin=triplet_margin,
            mining='semi-hard',
        )

    def forward(self, logits, features, labels):
        """logits: (B, num_classes), features: (B, D), labels: (B,)"""
        loss_ce = self.ce_loss(logits, labels)

        loss = self.ce_weight * loss_ce

        if self.supcon_weight > 0 and features.size(1) >= 32:
            loss_sc = self.supcon_loss(features, labels)
            loss = loss + self.supcon_weight * loss_sc
        else:
            loss_sc = torch.tensor(0.0)

        if self.triplet_weight > 0 and features.size(1) >= 32:
            loss_tr = self.triplet_loss(features, labels)
            loss = loss + self.triplet_weight * loss_tr
        else:
            loss_tr = torch.tensor(0.0)

        return loss, {
            'ce': loss_ce.item(),
            'supcon': loss_sc.item() if isinstance(loss_sc, torch.Tensor) else loss_sc,
            'triplet': loss_tr.item() if isinstance(loss_tr, torch.Tensor) else loss_tr,
        }


def create_loss(loss_name='focal', num_classes=70, class_weights=None, **kwargs):
    """损失函数工厂。

    可用:
      - focal: FocalLoss
      - ce: CrossEntropyLoss with label smoothing
      - supcon: SupervisedContrastiveLoss
      - triplet: TripletLoss
      - combined: CombinedLoss (分类+对比+三元组)
    """
    if loss_name == 'focal':
        return FocalLoss(
            weight=class_weights,
            gamma=kwargs.get('gamma', 2.0),
            label_smoothing=kwargs.get('label_smoothing', 0.05),
        )
    elif loss_name == 'ce':
        return LabelSmoothingCrossEntropy(
            smoothing=kwargs.get('label_smoothing', 0.1),
            weight=class_weights,
        )
    elif loss_name == 'supcon':
        return SupervisedContrastiveLoss(
            temperature=kwargs.get('temperature', 0.1),
        )
    elif loss_name == 'triplet':
        return TripletLoss(
            margin=kwargs.get('margin', 0.3),
            mining=kwargs.get('mining', 'semi-hard'),
        )
    elif loss_name == 'combined':
        return CombinedLoss(
            num_classes=num_classes, class_weights=class_weights,
            ce_weight=kwargs.get('ce_weight', 1.0),
            supcon_weight=kwargs.get('supcon_weight', 0.1),
            triplet_weight=kwargs.get('triplet_weight', 0.05),
            focal_gamma=kwargs.get('gamma', 2.0),
            label_smoothing=kwargs.get('label_smoothing', 0.05),
        )
    raise ValueError(f"Unknown loss: {loss_name}")


if __name__ == '__main__':
    print("Testing loss functions...\n")

    B, C, D = 32, 14, 256
    logits = torch.randn(B, C)
    features = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))
    weights = torch.ones(C)

    for name in ['focal', 'ce', 'combined']:
        loss_fn = create_loss(name, num_classes=C, class_weights=weights)
        if name == 'combined':
            loss, details = loss_fn(logits, features, labels)
            print(f"  {name:<12s} loss={loss.item():.4f}  "
                  f"ce={details['ce']:.4f}  sc={details['supcon']:.4f}  "
                  f"tr={details['triplet']:.4f}")
        else:
            loss = loss_fn(logits, labels)
            print(f"  {name:<12s} loss={loss.item():.4f}")

    # SupCon / Triplet (仅需 features)
    for name in ['supcon', 'triplet']:
        loss_fn = create_loss(name)
        loss = loss_fn(features, labels)
        print(f"  {name:<12s} loss={loss.item():.4f}")

    print("\nAll losses OK!")
