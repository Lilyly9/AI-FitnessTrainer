import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == "__main__":
    model = Gesture1DCNN(input_channels=6, num_classes=5)
    dummy = torch.randn(2, 6, 200)
    out = model(dummy)
    print(f"输出形状: {out.shape}")
