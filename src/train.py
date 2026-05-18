import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import Gesture1DCNN
from dataset import GestureDataset
from data_utils import NUM_CLASSES, CLASS_NAMES
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BATCH_SIZE = 32
EPOCHS = 80
LR = 0.01
MOMENTUM = 0.9
VAL_SPLIT = 0.2
MIXUP_ALPHA = 0.7
PATIENCE = 30
DATA_DIR = 'data/processed/'
NUM_ENSEMBLE = 3
# 类别权重 — 前 5 个 Gym Gesture 类别用经验权重，其余用 1.0
_gym_weights = torch.tensor([3.5, 2.5, 0.8, 0.5, 1.0], dtype=torch.float32)
if NUM_CLASSES <= 5:
    CLASS_WEIGHTS = _gym_weights[:NUM_CLASSES]
else:
    CLASS_WEIGHTS = torch.cat([_gym_weights, torch.ones(NUM_CLASSES - 5)])


def mixup_batch(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_model(seed):
    set_seed(seed)

    train_ds_aug = GestureDataset( # 训练用，增强
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'), train=True)
    train_ds_noaug = GestureDataset( # 验证用，不增强
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'), train=False)

    gen = torch.Generator().manual_seed(seed)
    num_val = int(len(train_ds_aug) * VAL_SPLIT)
    num_train = len(train_ds_aug) - num_val
    train_ds, _ = random_split(train_ds_aug, [num_train, num_val], generator=gen)
    _, val_ds = random_split(train_ds_noaug, [num_train, num_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Gesture1DCNN(input_channels=6, num_classes=NUM_CLASSES).to(device)

    class_weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=1e-3, nesterov=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

    best_acc = 0.0
    patience_counter = 0
    model_path = f'models/model_seed{seed}.pth'
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, y_a, y_b, lam = mixup_batch(inputs, labels, MIXUP_ALPHA)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_loss(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total

        avg_loss = train_loss / num_train
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f'  Seed {seed}: 早停于 Epoch {epoch}/{EPOCHS} (最佳 {best_acc:.4f})')
            break

    if patience_counter < PATIENCE:
        print(f'  Seed {seed}: 最佳验证准确率 = {best_acc:.4f}')
    return model_path, best_acc, device, history


def ensemble_predict(models, device, loader):
    for m in models:
        m.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = torch.zeros(inputs.size(0), NUM_CLASSES).to(device)
            for m in models:
                logits += m(inputs)
            logits /= len(models)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


# ===== 主流程 =====
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

t_start = time.time()

model_paths, best_accs = [], []
all_histories = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for seed in range(42, 42 + NUM_ENSEMBLE):
    t0 = time.time()
    print(f'\n--- 训练模型 {seed - 41}/{NUM_ENSEMBLE} (seed={seed}) ---')
    path, acc, device, hist = train_one_model(seed)
    model_paths.append(path)
    best_accs.append(acc)
    all_histories.append(hist)
    print(f'  耗时: {time.time() - t0:.1f}s')

t_train = time.time() - t_start
print(f'\n训练总耗时: {t_train:.1f}s ({t_train/60:.1f}min)')
print(f'平均验证准确率: {np.mean(best_accs):.4f} +/- {np.std(best_accs):.4f}')

# 加载集成模型
ensemble_models = []
for path in model_paths:
    m = Gesture1DCNN(input_channels=6, num_classes=NUM_CLASSES).to(device)
    m.load_state_dict(torch.load(path))
    ensemble_models.append(m)

# 测试集评估
test_ds = GestureDataset(
    os.path.join(DATA_DIR, 'x_test.npy'),
    os.path.join(DATA_DIR, 'y_test.npy'), train=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_labels = ensemble_predict(ensemble_models, device, test_loader)
test_acc = np.mean(all_preds == all_labels)
print(f'\n集成模型测试集准确率 (受试者5): {test_acc:.4f}')

for i in range(NUM_ENSEMBLE):
    preds, _ = ensemble_predict([ensemble_models[i]], device, test_loader)
    print(f'  单模型 seed={42+i}: {np.mean(preds == all_labels):.4f}')

best_idx = np.argmax(best_accs)
torch.save(ensemble_models[best_idx].state_dict(), 'models/best_model.pth')

# 分类报告
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
print('\n分类报告 (集成):\n')
print(report)
with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title('Confusion Matrix (Ensemble - Subject 5)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 训练曲线（3个模型对比）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#2ecc71', '#3498db', '#e74c3c']
for i, hist in enumerate(all_histories):
    epochs = range(1, len(hist['train_loss']) + 1)
    ax1.plot(epochs, hist['train_loss'], color=colors[i], alpha=0.8,
             label=f'Model {i+1} (seed={42+i})')
    ax2.plot(epochs, hist['val_acc'], color=colors[i], alpha=0.8,
             label=f'Model {i+1} (seed={42+i})')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle('Training Curves (3-Model Ensemble)', fontsize=14)
fig.tight_layout()
plt.savefig('results/training_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'训练曲线已保存到 results/training_curve.png')
print(f'混淆矩阵已保存到 results/confusion_matrix.png')
print(f'分类报告已保存到 results/classification_report.txt')
print(f'\n总耗时: {time.time() - t_start:.1f}s')
