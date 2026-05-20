"""
Domain Adversarial 训练器 — 解决 RecoFit(forearm) → wrist 的 domain gap。

核心思路 (Ganin & Lempitsky, 2015):
  - 共享 backbone 提取特征
  - 任务头: 预测动作类别 (70 类)
  - 域分类头 + Gradient Reversal Layer: 预测 forearm vs wrist
  - 联合优化: λ_task * L_task - λ_domain * L_domain

GRL 在反向传播时将域分类梯度取反，迫使 backbone 学习
domain-invariant 特征，从而消除 forearm/wrist 传感器位置差异。

用法:
  from domain_trainer import DomainAdversarialTrainer
  trainer = DomainAdversarialTrainer(model, num_classes=70, num_domains=2)
  trainer.train_epoch(train_loader, domain_loader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Gradient Reversal Layer (GRL)
# ============================================================
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层：前向传播恒等，反向传播取负。"""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


# ============================================================
# Domain Classifier Head
# ============================================================
class DomainClassifier(nn.Module):
    """2 层 MLP：特征 → domain (forearm=0 / wrist=1)。"""

    def __init__(self, feature_dim, hidden_dim=256, num_domains=2, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)  # GRL: 反转梯度
        return self.fc(x)


# ============================================================
# Domain Adversarial 模型包装
# ============================================================
class DomainAdversarialModel(nn.Module):
    """将 backbone + 任务头 + 域分类头 打包。

    backbone: 特征提取器 (e.g., Gesture1DCNN without FC)
    task_head: 动作分类器 (FC layer)
    domain_head: 域分类器 (DomainClassifier)
    """

    def __init__(self, backbone, task_head, feature_dim, num_domains=2,
                 domain_hidden=256, domain_dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.task_head = task_head
        self.domain_head = DomainClassifier(
            feature_dim, hidden_dim=domain_hidden,
            num_domains=num_domains, dropout=domain_dropout)

    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        task_out = self.task_head(features)
        domain_out = self.domain_head(features, alpha)
        return task_out, domain_out

    def get_features(self, x):
        """仅提取特征，不经过分类头。"""
        return self.backbone(x)


# ============================================================
# 构建 DANN 模型 (基于 Gesture1DCNN)
# ============================================================
def build_dann_model(num_classes=70, num_domains=2, dropout=0.5):
    """基于 Gesture1DCNN 构建 domain adversarial 模型。

    Gesture1DCNN backbone: 3 层 Conv1d + InstanceNorm → GlobalPool(avg+max)
    → 256-dim feature
    Task head: Linear(256 → num_classes)
    Domain head: DomainClassifier(256 → 128 → 2)
    """
    from model_v2 import Gesture1DCNN

    # 用 Gesture1DCNN 作为 backbone
    base = Gesture1DCNN(input_channels=6, num_classes=num_classes, dropout=dropout)

    # 分离 backbone (到 global pool 之前) 和 task head (FC)
    feature_dim = 256  # Gesture1DCNN: 128*2 from avg+max pool

    backbone = nn.Sequential(
        base.conv1, base.in1, nn.ReLU(), base.pool1, base.drop1,
        base.conv2, base.in2, nn.ReLU(), base.pool2, base.drop2,
        base.conv3, base.in3, nn.ReLU(), base.pool3, base.drop3,
    )

    # task_head: pool + fc
    task_head = nn.Sequential(
        base.gap,  # AdaptiveAvgPool1d(1)
        nn.Flatten(),  # We need to handle both avg and max pool
    )

    # Actually, the original Gesture1DCNN forward concatenates gap+gmp then fc.
    # Let's define clean feature extraction.
    class FeatureExtractor(nn.Module):
        def __init__(self, conv_layers):
            super().__init__()
            self.conv = conv_layers
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.gmp = nn.AdaptiveMaxPool1d(1)

        def forward(self, x):
            x = self.conv(x)
            return torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1)

    feature_extractor = FeatureExtractor(nn.Sequential(
        base.conv1, base.in1, nn.ReLU(), base.pool1, base.drop1,
        base.conv2, base.in2, nn.ReLU(), base.pool2, base.drop2,
        base.conv3, base.in3, nn.ReLU(), base.pool3, base.drop3,
    ))

    task_head = base.fc  # Linear(256, num_classes)

    return DomainAdversarialModel(
        backbone=feature_extractor,
        task_head=task_head,
        feature_dim=feature_dim,
        num_domains=num_domains,
    )


# ============================================================
# Domain Adversarial 训练器
# ============================================================
class DomainAdversarialTrainer:
    """封装 DANN 训练逻辑。

    训练参数:
      domain_lambda: 域损失权重 (默认 0.1，从 0 → 1 渐进增加)
      alpha_schedule: GRL alpha 渐进策略
        'fixed': alpha=1.0
        'ramp': alpha = 2/(1+exp(-10*p)) - 1, 其中 p=epoch/total_epochs
    """

    def __init__(self, model, num_classes, num_domains=2,
                 domain_lambda=0.1, alpha_schedule='ramp',
                 task_criterion=None, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.num_domains = num_domains
        self.domain_lambda = domain_lambda
        self.alpha_schedule = alpha_schedule

        # 任务损失：CrossEntropy with class weights
        self.task_criterion = task_criterion or nn.CrossEntropyLoss()

        # 域损失：CrossEntropy (forearm vs wrist 通常均衡)
        self.domain_criterion = nn.CrossEntropyLoss()

    def compute_alpha(self, epoch, total_epochs):
        """计算 GRL 的 alpha 系数。渐进增加使训练初期聚焦任务学习。"""
        if self.alpha_schedule == 'fixed':
            return 1.0
        elif self.alpha_schedule == 'ramp':
            p = epoch / max(total_epochs, 1)
            return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        else:
            return 1.0

    def train_step(self, x_task, y_task, x_domain, y_domain, epoch, total_epochs,
                   optimizer):
        """单步训练：任务损失 + 域对抗损失。

        Args:
          x_task: (B, 6, 200) 有标签动作数据
          y_task: (B,) 动作类别标签
          x_domain: (B, 6, 200) 域标签数据 (forearm/wrist)
          y_domain: (B,) 域标签 (0=forearm, 1=wrist)
        """
        self.model.train()

        alpha = self.compute_alpha(epoch, total_epochs)

        # --- 任务损失 ---
        task_out, domain_out = self.model(x_task, alpha=alpha)
        task_loss = self.task_criterion(task_out, y_task)

        # --- 域对抗损失 ---
        domain_loss = self.domain_criterion(domain_out, y_domain)

        # --- 联合损失 ---
        total_loss = task_loss + self.domain_lambda * domain_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 准确率
        task_acc = (task_out.argmax(1) == y_task).float().mean()
        domain_acc = (domain_out.argmax(1) == y_domain).float().mean()

        return {
            'loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'domain_loss': domain_loss.item(),
            'task_acc': task_acc.item(),
            'domain_acc': domain_acc.item(),
            'alpha': alpha,
        }

    @torch.no_grad()
    def evaluate(self, loader, return_features=False):
        """评估：仅任务准确率（不需要域标签）。"""
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        all_features = [] if return_features else None

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            task_out, _ = self.model(x, alpha=0)  # alpha=0 → GRL 关闭
            pred = task_out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            if return_features:
                features = self.model.get_features(x)
                all_features.append(features.cpu().numpy())

        acc = correct / max(total, 1)
        result = {'acc': acc, 'preds': np.array(all_preds), 'labels': np.array(all_labels)}
        if return_features:
            result['features'] = np.concatenate(all_features, axis=0)
        return result


# ============================================================
# 辅助：创建域数据加载器
# ============================================================
def create_domain_loaders(x_train, y_train, d_train, x_test, y_test, d_test,
                          batch_size=64):
    """从合并数据创建任务数据加载器和域数据加载器。

    域数据加载器额外包含域标签，用于域对抗训练。
    """
    train_task_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))
    test_task_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long))

    # 域数据：用 domain 标签作为目标
    train_domain_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(d_train, dtype=torch.long))
    test_domain_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(d_test, dtype=torch.long))

    train_task_loader = DataLoader(train_task_ds, batch_size=batch_size, shuffle=True)
    test_task_loader = DataLoader(test_task_ds, batch_size=batch_size, shuffle=False)
    train_domain_loader = DataLoader(train_domain_ds, batch_size=batch_size, shuffle=True)

    return train_task_loader, test_task_loader, train_domain_loader


# ============================================================
# 完整训练循环
# ============================================================
def train_dann(model, train_task_loader, train_domain_loader, test_loader,
               num_classes, epochs=50, lr=0.01, domain_lambda=0.1,
               device='cpu', save_path='models/dann_model.pth'):
    """完整的 DANN 训练循环。

    Returns:
      best_acc, history dict
    """
    from sklearn.metrics import f1_score

    trainer = DomainAdversarialTrainer(
        model, num_classes=num_classes, num_domains=2,
        domain_lambda=domain_lambda, device=device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=1e-3, nesterov=True)

    best_acc = 0
    history = {'train_loss': [], 'test_acc': [], 'test_mf1': []}

    for epoch in range(1, epochs + 1):
        # 同时迭代任务数据和域数据
        domain_iter = iter(train_domain_loader)

        epoch_losses = []
        for (x_task, y_task) in train_task_loader:
            try:
                x_domain, y_domain = next(domain_iter)
            except StopIteration:
                domain_iter = iter(train_domain_loader)
                x_domain, y_domain = next(domain_iter)

            x_task, y_task = x_task.to(device), y_task.to(device)
            x_domain, y_domain = x_domain.to(device), y_domain.to(device)

            metrics = trainer.train_step(
                x_task, y_task, x_domain, y_domain,
                epoch, epochs, optimizer)
            epoch_losses.append(metrics['loss'])

        # 评估
        result = trainer.evaluate(test_loader)
        mf1 = f1_score(result['labels'], result['preds'], average='macro', zero_division=0)

        history['train_loss'].append(np.mean(epoch_losses))
        history['test_acc'].append(result['acc'])
        history['test_mf1'].append(mf1)

        if result['acc'] > best_acc:
            best_acc = result['acc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch, 'acc': best_acc, 'mf1': mf1,
                'num_classes': num_classes,
            }, save_path)

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            print(f"Ep {epoch:3d} | loss={np.mean(epoch_losses):.4f} "
                  f"| acc={result['acc']:.4f} | mf1={mf1:.4f} "
                  f"| alpha={metrics['alpha']:.3f}")

    print(f"\nBest test acc: {best_acc:.4f}")
    return best_acc, history


if __name__ == '__main__':
    print("Domain Adversarial Training module loaded.")
    print("Usage:")
    print("  from domain_trainer import build_dann_model, train_dann, create_domain_loaders")
    print("  model = build_dann_model(num_classes=70)")
    print("  train_dann(model, train_loader, domain_loader, test_loader, num_classes=70)")
