import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Gesture1DCNN
import numpy as np
import os
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
VAL_SPLIT = 0.2
DATA_DIR = 'data/processed/'
NUM_CLASSES = 4


# 加载数据
x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy')).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy')).astype(np.int64)
x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy')).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy')).astype(np.int64)

# 转为张量
x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t = torch.tensor(x_test)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 划分训练/验证集
full_train = TensorDataset(x_train_t, y_train_t)
num_val = int(len(full_train) * VAL_SPLIT)
num_train = len(full_train) - num_val
train_ds, val_ds = random_split(full_train, [num_train, num_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# 设备、模型、损失、优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Gesture1DCNN(input_channels=6, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练循环
best_acc = 0.0
os.makedirs('models', exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    val_acc = val_correct / val_total

    print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss/num_train:.4f} | Val Acc: {val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'  => 保存最佳模型 (Val Acc: {val_acc:.4f})')

print('训练完成')

# 测试最佳模型
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()
test_acc = test_correct / test_total
print(f'测试集准确率: {test_acc:.4f}')