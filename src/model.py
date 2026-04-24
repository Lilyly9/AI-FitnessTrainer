import torch
import torch.nn as nn
import torch.nn.functional as F

class Gesture1DCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5):
        super(Gesture1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = Gesture1DCNN(input_channels=6, num_classes=4)
    dummy = torch.randn(2, 6, 200)
    out = model(dummy)
    print(f"输出形状: {out.shape}")   # 期望 torch.Size([2, 5])