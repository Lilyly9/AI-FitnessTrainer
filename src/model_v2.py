"""
改进版模型架构 — 专为多数据集 IMU 动作识别设计。

包含:
  - ResCNN1D: 残差1D CNN，BatchNorm版（训练稳定、收敛快）
  - DeepConvLSTM: HAR经典结构 (Ordonez & Roggen, 2016)
  - CNNTransformer: CNN stem + Transformer encoder
  - Gesture1DCNN: 原始模型（向后兼容）

所有模型统一接口: (batch, 6, window_size) -> (batch, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# ResBlock1D — BatchNorm 残差块
# ============================================================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + self.shortcut(x))


class ResCNN1D(nn.Module):
    """残差 CNN — BatchNorm，IMU分类主力模型。

    结构: stem(k=15,s=2) -> 3组残差块 -> adaptive pool -> FC
    参数量约 1.55M，适合 70 类 IMU 动作分类。
    """

    def __init__(self, input_channels=6, num_classes=70, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer2 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * 2, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, dropout):
        layers = [ResBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def _extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_avg = self.avgpool(x).flatten(1)
        x_max = self.maxpool(x).flatten(1)
        return torch.cat([x_avg, x_max], dim=1)

    def forward(self, x, return_features=False):
        features = self._extract_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, features, None
        return logits

    def get_features(self, x):
        return self._extract_features(x)


# ============================================================
# DeepConvLSTM — HAR 领域经典模型
# ============================================================
class DeepConvLSTM(nn.Module):
    """3层CNN + 2层BiLSTM + FC分类头"""

    def __init__(self, input_channels=6, num_classes=70,
                 conv_channels=(64, 128, 256), lstm_hidden=256,
                 lstm_layers=2, dropout=0.4):
        super().__init__()
        c1, c2, c3 = conv_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, c1, 7, padding=3), nn.BatchNorm1d(c1),
            nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(c1, c2, 5, padding=2), nn.BatchNorm1d(c2),
            nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(c2, c3, 3, padding=1), nn.BatchNorm1d(c3),
            nn.ReLU(), nn.MaxPool1d(2))
        self.drop_cnn = nn.Dropout(dropout * 0.7)

        self.lstm = nn.LSTM(c3, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.drop_lstm = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def _extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop_cnn(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.drop_lstm(out[:, -1, :])

    def forward(self, x, return_features=False):
        features = self._extract_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, features, None
        return logits

    def get_features(self, x):
        return self._extract_features(x)


# ============================================================
# Gesture1DCNN — 原始模型（向后兼容）
# ============================================================
class Gesture1DCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 11, padding=5)
        self.in1 = nn.InstanceNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.in2 = nn.InstanceNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.in3 = nn.InstanceNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _extract_features(self, x):
        x = self.pool1(F.relu(self.in1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.in2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.in3(self.conv3(x))))
        x = self.drop3(x)
        return torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1)

    def forward(self, x, return_features=False):
        features = self._extract_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, features, None
        return logits

    def get_features(self, x):
        return self._extract_features(x)


# ============================================================
# 模型工厂
# ============================================================
def create_model_v2(model_name='ResCNN1D', input_channels=6, num_classes=70,
                    dropout=0.3, **kwargs):
    """v2 + v3 统一模型工厂 — 向后兼容。

    v2 模型: ResCNN1D, DeepConvLSTM, Gesture1DCNN
    v3 模型: TCN, AttnConvLSTM, CNNTransformerV2, MultiStreamCNN
    """
    # v2 模型
    if model_name == 'ResCNN1D':
        return ResCNN1D(input_channels=input_channels, num_classes=num_classes,
                        dropout=dropout)
    elif model_name == 'DeepConvLSTM':
        return DeepConvLSTM(input_channels=input_channels, num_classes=num_classes,
                            dropout=dropout, **kwargs)
    elif model_name == 'Gesture1DCNN':
        return Gesture1DCNN(input_channels=input_channels, num_classes=num_classes,
                            dropout=dropout)

    # v3 模型 — 自动导入
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model_v3 import create_model_v3
        return create_model_v3(model_name, input_channels=input_channels,
                              num_classes=num_classes, dropout=dropout, **kwargs)
    except ImportError:
        pass

    raise ValueError(f"Unknown model: {model_name}. "
                     f"Available: ResCNN1D, DeepConvLSTM, Gesture1DCNN, "
                     f"TCN, AttnConvLSTM, CNNTransformerV2, MultiStreamCNN")


if __name__ == '__main__':
    for name in ['ResCNN1D', 'DeepConvLSTM', 'Gesture1DCNN']:
        m = create_model_v2(model_name=name, num_classes=70)
        d = torch.randn(2, 6, 200)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name}: output={m(d).shape}, params={params:,}")
