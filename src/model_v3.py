"""
增强模型架构 v3 — 针对 IMU 时序动作识别的深度优化。

新增架构:
  - TCN: Temporal Convolutional Network (dilated conv, 大感受野)
  - AttnConvLSTM: DeepConvLSTM + 多头自注意力池化
  - CNNTransformerV2: CNN stem + Transformer + 可学习注意力池化
  - MultiStreamCNN: 多尺度流 + 时序注意力融合

所有模型统一接口: (batch, C, T) -> (batch, num_classes)
支持 feature extraction 模式用于度量学习与可视化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 基础组件
# ============================================================
class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + Dropout 基础块。"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1,
                 dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """带膨胀卷积的残差块 — TCN 基础单元。"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Sequential()
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + residual)


# ============================================================
# 1) TCN — Temporal Convolutional Network
# ============================================================
class TCN(nn.Module):
    """时序卷积网络: 堆叠膨胀残差块，指数级增大感受野。

    感受野计算: 每层 kernel_size=7, dilation=2^i
    8 层: RF ≈ 1 + 2*(7-1)*(1+2+4+...+128) = 1+12*255 = 3061 >> 200 ✓
    """

    def __init__(self, input_channels=6, num_classes=70,
                 hidden_channels=64, levels=8, kernel_size=7, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = input_channels

        for i in range(levels):
            dilation = 2 ** i
            out_ch = hidden_channels * (2 ** min(i, 2))  # 64→128→256→256...
            layers.append(ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.final_ch = in_ch

        # 注意力池化替代简单 global pool
        self.attn_pool = AttentionPool1D(in_ch, pool_dim=128)

        self.fc = nn.Sequential(
            nn.Linear(in_ch * 2, 256),  # attn_pool concat
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        feat = self.network(x)  # (B, C', T')
        pooled, attn_weights = self.attn_pool(feat, return_weights=True)
        # concat attn_pool + max_pool 保留极端值信息
        max_pooled = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        combined = torch.cat([pooled, max_pooled], dim=1)
        logits = self.fc(combined)
        if return_features:
            return logits, combined, attn_weights
        return logits

    def get_features(self, x):
        feat = self.network(x)
        pooled, _ = self.attn_pool(feat, return_weights=True)
        max_pooled = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        return torch.cat([pooled, max_pooled], dim=1)


# ============================================================
# 2) AttnConvLSTM — DeepConvLSTM + 注意力池化
# ============================================================
class AttnConvLSTM(nn.Module):
    """CNN 降采样 → 多层 BiLSTM → 多头自注意力池化 → FC。

    相比原始 DeepConvLSTM (只用最后隐状态):
    - 自注意力池化利用所有时间步的隐状态
    - 残差连接稳定训练
    """

    def __init__(self, input_channels=6, num_classes=70,
                 conv_channels=(64, 128, 256), lstm_hidden=256,
                 lstm_layers=2, dropout=0.4):
        super().__init__()
        c1, c2, c3 = conv_channels

        self.conv1 = ConvBlock(input_channels, c1, 7, dropout=dropout * 0.5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = ConvBlock(c1, c2, 5, dropout=dropout * 0.5)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = ConvBlock(c2, c3, 3, dropout=dropout * 0.5)
        self.pool3 = nn.MaxPool1d(2)
        self.drop_cnn = nn.Dropout(dropout * 0.7)

        self.lstm = nn.LSTM(c3, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.drop_lstm = nn.Dropout(dropout)

        lstm_out = lstm_hidden * 2  # bidirectional

        # 多头自注意力池化
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=4, dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attn_query = nn.Parameter(torch.randn(1, 1, lstm_out) * 0.02)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        # CNN 降采样
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.drop_cnn(x).transpose(1, 2)  # (B, T', c3)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, T', lstm_hidden*2)
        lstm_out = self.drop_lstm(lstm_out)

        # 注意力池化: query 对 LSTM 输出做 cross-attention
        query = self.attn_query.expand(x.size(0), -1, -1)  # (B, 1, D)
        attn_out, attn_weights = self.attn_pool(query, lstm_out, lstm_out)
        features = attn_out.squeeze(1)  # (B, D)

        logits = self.fc(features)
        if return_features:
            return logits, features, attn_weights.squeeze(1)
        return logits

    def get_features(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.drop_cnn(x).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        query = self.attn_query.expand(x.size(0), -1, -1)
        attn_out, _ = self.attn_pool(query, lstm_out, lstm_out)
        return attn_out.squeeze(1)


# ============================================================
# 3) CNNTransformerV2 — 改进版 CNN + Transformer
# ============================================================
class PositionalEncoding(nn.Module):
    """正弦位置编码。"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class AttentionPool1D(nn.Module):
    """可学习注意力池化: 自动学习哪些时间步更重要。

    score = softmax(W * x + b), output = sum(score * x)
    """
    def __init__(self, in_dim, pool_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, pool_dim),
            nn.Tanh(),
            nn.Linear(pool_dim, 1),
        )

    def forward(self, x, return_weights=False):
        """x: (B, C, T) → output: (B, C), weights: (B, T)"""
        x_t = x.transpose(1, 2)  # (B, T, C)
        scores = self.attn(x_t).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x_t).squeeze(1)  # (B, C)
        if return_weights:
            return pooled, weights
        return pooled


class CNNTransformerV2(nn.Module):
    """CNN stem → PositionalEncoding → Transformer → 混合注意力池化。

    改进点 vs GestureCNNTransformer:
      - 可学习注意力池化 + adaptive pool (保留全局统计)
      - Pre-LN Transformer (训练更稳定)
      - 更大的 CNN stem 输出维度
    """

    def __init__(self, input_channels=6, num_classes=70,
                 d_model=192, nhead=6, num_layers=3,
                 dim_feedforward=384, dropout=0.3):
        super().__init__()

        # CNN stem: 更大的输出维度
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 48, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(48), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(48, 96, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96, d_model, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=200, dropout=dropout)

        # Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # 混合池化: 注意力 + 统计
        self.attn_pool = AttentionPool1D(d_model, pool_dim=96)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        fc_in = d_model * 3  # attn + avg + max
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        # CNN stem
        x = self.stem(x)  # (B, d_model, T')
        x = x.transpose(1, 2)  # (B, T', d_model)

        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (B, T', d_model)
        x = x.transpose(1, 2)  # (B, d_model, T')

        # 混合池化
        attn_pooled, attn_weights = self.attn_pool(x, return_weights=True)
        avg_pooled = self.avg_pool(x).squeeze(-1)
        max_pooled = self.max_pool(x).squeeze(-1)
        features = torch.cat([attn_pooled, avg_pooled, max_pooled], dim=1)

        logits = self.fc(features)
        if return_features:
            return logits, features, attn_weights
        return logits

    def get_features(self, x):
        x = self.stem(x).transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x).transpose(1, 2)
        attn_pooled, _ = self.attn_pool(x, return_weights=True)
        avg_pooled = self.avg_pool(x).squeeze(-1)
        max_pooled = self.max_pool(x).squeeze(-1)
        return torch.cat([attn_pooled, avg_pooled, max_pooled], dim=1)


# ============================================================
# 4) MultiStreamCNN — 多尺度流 + 门控融合
# ============================================================
class GatedFusion(nn.Module):
    """门控多尺度融合: 学习不同尺度特征的重要性权重。"""
    def __init__(self, dims):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(sum(dims), len(dims) * 64),
            nn.ReLU(),
            nn.Linear(len(dims) * 64, len(dims)),
            nn.Sigmoid(),
        )

    def forward(self, *features):
        combined = torch.cat(features, dim=1)
        gates = self.gate(combined)  # (B, num_streams)
        # 按通道加权
        outputs = []
        start = 0
        for i, feat in enumerate(features):
            outputs.append(feat * gates[:, i:i+1])
        return torch.cat(outputs, dim=1)


class MultiStreamCNN(nn.Module):
    """三流多尺度 CNN: 小核(高频) + 中核(动作) + 大核(周期)。

    每个流独立处理不同时间尺度的模式，最后门控融合。
    """

    def __init__(self, input_channels=6, num_classes=70, dropout=0.3):
        super().__init__()

        # 流 1: 小核 — 捕捉快速变化
        self.stream_small = nn.Sequential(
            ConvBlock(input_channels, 64, 5, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 5, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBlock(128, 128, 3, dropout=dropout),
        )

        # 流 2: 中核 — 捕捉动作级模式
        self.stream_med = nn.Sequential(
            ConvBlock(input_channels, 64, 11, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 9, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBlock(128, 128, 7, dropout=dropout),
        )

        # 流 3: 大核 — 捕捉完整动作周期
        self.stream_large = nn.Sequential(
            ConvBlock(input_channels, 64, 21, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 15, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBlock(128, 128, 11, dropout=dropout),
        )

        # 每个流独立池化
        self.pool_small = AttentionPool1D(128, 64)
        self.pool_med = AttentionPool1D(128, 64)
        self.pool_large = AttentionPool1D(128, 64)

        # 门控融合
        self.fusion = GatedFusion([128, 128, 128])

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        f_small = self.stream_small(x)
        f_med = self.stream_med(x)
        f_large = self.stream_large(x)

        p_small, w_small = self.pool_small(f_small, return_weights=True)
        p_med, w_med = self.pool_med(f_med, return_weights=True)
        p_large, w_large = self.pool_large(f_large, return_weights=True)

        features = self.fusion(p_small, p_med, p_large)
        logits = self.fc(features)

        if return_features:
            # 返回各流的注意力权重用于可视化
            attn_weights = {
                'small': w_small, 'med': w_med, 'large': w_large,
            }
            return logits, features, attn_weights
        return logits

    def get_features(self, x):
        f_small = self.stream_small(x)
        f_med = self.stream_med(x)
        f_large = self.stream_large(x)
        p_small = self.pool_small(f_small)
        p_med = self.pool_med(f_med)
        p_large = self.pool_large(f_large)
        return self.fusion(p_small, p_med, p_large)


# ============================================================
# 模型工厂
# ============================================================
def create_model_v3(model_name='TCN', input_channels=6, num_classes=70,
                    dropout=0.3, **kwargs):
    """v3 模型工厂。

    可用模型:
      - TCN: Temporal Convolutional Network
      - AttnConvLSTM: DeepConvLSTM + Multihead Attention Pooling
      - CNNTransformerV2: CNN + Transformer + 混合池化
      - MultiStreamCNN: 三流多尺度 + 门控融合
    """
    if model_name == 'TCN':
        return TCN(input_channels=input_channels, num_classes=num_classes,
                   dropout=dropout,
                   hidden_channels=kwargs.get('hidden_channels', 64),
                   levels=kwargs.get('levels', 7),
                   kernel_size=kwargs.get('kernel_size', 7))
    elif model_name == 'AttnConvLSTM':
        return AttnConvLSTM(input_channels=input_channels, num_classes=num_classes,
                           dropout=dropout,
                           conv_channels=kwargs.get('conv_channels', (64, 128, 256)),
                           lstm_hidden=kwargs.get('lstm_hidden', 256),
                           lstm_layers=kwargs.get('lstm_layers', 2))
    elif model_name == 'CNNTransformerV2':
        return CNNTransformerV2(input_channels=input_channels, num_classes=num_classes,
                                dropout=dropout,
                                d_model=kwargs.get('d_model', 192),
                                nhead=kwargs.get('nhead', 6),
                                num_layers=kwargs.get('num_layers', 3),
                                dim_feedforward=kwargs.get('dim_feedforward', 384))
    elif model_name == 'MultiStreamCNN':
        return MultiStreamCNN(input_channels=input_channels, num_classes=num_classes,
                             dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}. "
                     f"Choose from: TCN, AttnConvLSTM, CNNTransformerV2, MultiStreamCNN")


if __name__ == '__main__':
    print("Testing v3 models...\n")
    dummy = torch.randn(2, 6, 200)
    for name in ['TCN', 'AttnConvLSTM', 'CNNTransformerV2', 'MultiStreamCNN']:
        model = create_model_v3(name, num_classes=14, dropout=0.3)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 测试 forward, get_features, return_features
        out = model(dummy)
        feats = model.get_features(dummy)
        logits, features, attn = model(dummy, return_features=True)

        print(f"  {name:<20s}  output={str(out.shape):<20s}  "
              f"features={str(features.shape):<20s}  params={params:>8,}")
    print("\nAll v3 models OK!")
