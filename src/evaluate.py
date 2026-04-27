import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from model import Gesture1DCNN
import os

DATA_DIR = 'data/processed/'
MODEL_PATH = 'models/best_model.pth'
NUM_CLASSES = 4   # 改为 4
BATCH_SIZE = 32

# 加载测试数据
x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy')).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy')).astype(np.int64)
x_test_t = torch.tensor(x_test)
y_test_t = torch.tensor(y_test, dtype=torch.long)
test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Gesture1DCNN(input_channels=6, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 收集预测结果
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算指标
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
rec = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# 分类报告
class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
print("\n分类报告:\n", report)

# 保存分类报告文本
os.makedirs('results', exist_ok=True)
with open('results/classification_report.txt', 'w') as f:
    f.write(report)

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.savefig('results/confusion_matrix.png')
print("混淆矩阵:\n", cm)
print("混淆矩阵已保存到内存")