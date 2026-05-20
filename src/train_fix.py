"""修正版训练 — 移除 WeightedSampler，使用 class weights + 长训练。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, json, time
from model_v2 import create_model_v2
from dataset import GestureDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import functools
print = functools.partial(print, flush=True)
LOG = open('results/train_log_v3.txt', 'w', encoding='utf-8', buffering=1)

def log(msg):
    print(msg)
    LOG.write(msg + '\n')
    LOG.flush()

log("=== Corrected Training v3 (No WeightedSampler) ===")

# Meta
with open('data/processed/dataset_meta.json') as f:
    meta = json.load(f)
n_cls = meta['num_classes']; cls_names = meta['class_names']
log(f"Classes: {n_cls}")

# Data
x_train = np.load('data/processed/x_train.npy').astype(np.float32)
y_train = np.load('data/processed/y_train.npy').astype(np.int64).flatten()
x_test = np.load('data/processed/x_test.npy').astype(np.float32)
y_test = np.load('data/processed/y_test.npy').astype(np.int64).flatten()
log(f"Train: {x_train.shape}  Test: {x_test.shape}")

# Stratified train/val
tr_idx, va_idx = train_test_split(
    np.arange(len(y_train)), test_size=0.2, random_state=42, stratify=y_train)

tr_labels = y_train[tr_idx]
cnt = Counter(tr_labels)

# Class weights for loss (NOT sampler)
cls_w = torch.zeros(n_cls)
for c in range(n_cls):
    cls_w[c] = len(tr_idx) / (n_cls * max(cnt.get(c, 1), 1))
cls_w = cls_w.clamp(0.1, 10.0)
log(f"Class weights: min={cls_w.min():.2f} max={cls_w.max():.2f}")

tr_ds = GestureDataset.from_arrays(x_train[tr_idx], y_train[tr_idx], train=True)
va_ds = GestureDataset.from_arrays(x_train[va_idx], y_train[va_idx], train=False)
te_ds = GestureDataset.from_arrays(x_test, y_test, train=False)

# NO WeightedRandomSampler — use shuffle=True
tr_ldr = DataLoader(tr_ds, batch_size=128, shuffle=True)
va_ldr = DataLoader(va_ds, batch_size=256, shuffle=False)
te_ldr = DataLoader(te_ds, batch_size=256, shuffle=False)

# Model
device = torch.device('cpu')
model = create_model_v2('ResCNN1D', num_classes=n_cls, dropout=0.3).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Model: {n_params:,} params")

criterion = nn.CrossEntropyLoss(weight=cls_w.to(device), label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            weight_decay=1e-3, nesterov=True)
cos = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)

WARMUP = 10
TOTAL = 100
t_start = time.time()
best_mf1 = 0.0
patience = 30
wait = 0

for ep in range(TOTAL):
    ep_t0 = time.time()

    # Warmup then cosine
    if ep < WARMUP:
        for g in optimizer.param_groups:
            g['lr'] = 0.01 * (ep + 1) / WARMUP
    else:
        cos.step(ep - WARMUP)

    # Train
    model.train()
    tloss, correct, total = 0.0, 0, 0
    for x, y in tr_ldr:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tloss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    # Validate
    model.eval()
    vp, vl = [], []
    with torch.no_grad():
        for x, y in va_ldr:
            x, y = x.to(device), y.to(device)
            vp.extend(model(x).argmax(1).tolist())
            vl.extend(y.tolist())
    vp = np.array(vp); vl = np.array(vl)
    mf1 = f1_score(vl, vp, average='macro', zero_division=0)

    lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - ep_t0
    mark = ''
    if mf1 > best_mf1:
        best_mf1 = mf1
        wait = 0
        torch.save({'model_state_dict': model.state_dict(),
                    'best_macro_f1': best_mf1, 'epoch': ep,
                    'model_name': 'ResCNN1D', 'num_classes': n_cls},
                   'models/best_model_v2.pth')
        mark = ' *'
    else:
        wait += 1

    log(f"Ep {ep+1:3d} | loss={tloss/total:.4f} | acc={correct/total:.4f} "
        f"| mf1={mf1:.4f} | lr={lr:.6f} | {elapsed:.0f}s{mark}")

    if wait >= patience:
        log(f"Early stop at ep {ep+1} (best mf1={best_mf1:.4f})")
        break

# Test best model
ckpt = torch.load('models/best_model_v2.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tp, tl = [], []
with torch.no_grad():
    for x, y in te_ldr:
        x, y = x.to(device), y.to(device)
        tp.extend(model(x).argmax(1).tolist())
        tl.extend(y.tolist())
tp = np.array(tp); tl = np.array(tl)
tacc = np.mean(tp == tl)
tmf1 = f1_score(tl, tp, average='macro', zero_division=0)

log(f"\n{'='*60}")
log(f"FINAL: Test Acc={tacc:.4f} ({tacc*n_cls:.1f}x baseline)  Macro F1={tmf1:.4f}")
log(f"Time={time.time()-t_start:.0f}s  Best epoch={ckpt['epoch']+1}")

# Per-class
rdict = classification_report(tl, tp, labels=list(range(n_cls)),
                              target_names=cls_names, output_dict=True, zero_division=0)
pcls = [(n, rdict[n]['f1-score'], rdict[n]['support'])
        for n in cls_names if n in rdict]
pcls.sort(key=lambda x: x[1], reverse=True)
log("\nTop 20 by F1:")
for n, f1, s in pcls[:20]:
    log(f"  f1={f1:.3f}  {n:<45s} n={s:.0f}")
nz = sum(1 for _, f1, _ in pcls if f1 == 0)
log(f"  ...  F1=0: {nz}/{n_cls}")

# Per-dataset
log("\nPer-dataset breakdown:")
if 'label_sets' in meta:
    for ds, lbls in meta['label_sets'].items():
        m = np.isin(tl, lbls)
        if m.sum():
            a = np.mean(tp[m] == tl[m])
            f = f1_score(tl[m], tp[m], labels=lbls, average='macro', zero_division=0)
            log(f"  {ds}: acc={a:.4f}  mf1={f:.4f}  n={m.sum()}")

log(f"\nModel saved: models/best_model_v2.pth")
LOG.close()
