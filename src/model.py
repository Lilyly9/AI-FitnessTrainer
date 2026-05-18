import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Gesture1DCNN(nn.Module):
    """1D CNN with InstanceNorm for subject-invariant features."""

    def __init__(self, input_channels=6, num_classes=5, dropout=0.5):
        super(Gesture1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=11, padding=5)
        self.in1 = nn.InstanceNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.in2 = nn.InstanceNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.in3 = nn.InstanceNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.in1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.in2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.in3(self.conv3(x))))
        x = self.dropout3(x)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x_max = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.fc(x)
        return x


# ============================================================
# 位置编码（Transformer 用）
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# CNN + Transformer 混合架构
# ============================================================
class GestureCNNTransformer(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, dropout=0.5,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.InstanceNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.InstanceNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.InstanceNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.conv_stem(x).transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x).transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# ============================================================
# 多尺度 CNN 架构
# ============================================================
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        ch_small = (out_ch + 2) // 3
        ch_med = (out_ch + 2) // 3
        ch_large = out_ch - ch_small - ch_med
        self.out_ch_actual = ch_small + ch_med + ch_large
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_ch, ch_small, kernel_size=5, padding=2),
            nn.InstanceNorm1d(ch_small), nn.ReLU())
        self.branch_med = nn.Sequential(
            nn.Conv1d(in_ch, ch_med, kernel_size=11, padding=5),
            nn.InstanceNorm1d(ch_med), nn.ReLU())
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_ch, ch_large, kernel_size=21, padding=10),
            nn.InstanceNorm1d(ch_large), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        s = self.branch_small(x)
        m = self.branch_med(x)
        l = self.branch_large(x)
        return self.dropout(self.pool(torch.cat([s, m, l], dim=1)))


class GestureMultiScaleCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, dropout=0.5):
        super().__init__()
        self.block1 = MultiScaleConvBlock(input_channels, 96, dropout)
        self.block2 = MultiScaleConvBlock(self.block1.out_ch_actual, 192, dropout)
        self.block3 = MultiScaleConvBlock(self.block2.out_ch_actual, 256, dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(self.block3.out_ch_actual * 2, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x_max = self.global_max_pool(x).squeeze(-1)
        return self.fc(torch.cat([x_avg, x_max], dim=1))


def create_model(model_name='Gesture1DCNN', input_channels=6, num_classes=5, dropout=0.5,
                 transformer_cfg=None, **kwargs):
    if model_name == 'Gesture1DCNN':
        return Gesture1DCNN(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    elif model_name == 'GestureCNNTransformer':
        cfg = transformer_cfg or {}
        return GestureCNNTransformer(
            input_channels=input_channels, num_classes=num_classes, dropout=dropout,
            d_model=cfg.get('d_model', 128), nhead=cfg.get('nhead', 4),
            num_layers=cfg.get('num_layers', 2),
            dim_feedforward=cfg.get('dim_feedforward', 256))
    elif model_name == 'GestureMultiScaleCNN':
        return GestureMultiScaleCNN(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"未知模型: {model_name}")


if __name__ == "__main__":
    for name in ['Gesture1DCNN', 'GestureCNNTransformer', 'GestureMultiScaleCNN']:
        model = create_model(model_name=name, input_channels=6, num_classes=5)
        dummy = torch.randn(2, 6, 200)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: 输出={model(dummy).shape}, 参数量={params:,}")
