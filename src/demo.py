"""
模型推理演示 — 支持全部架构、全部类别数。

自动发现模型路径，优先级:
  1. models/ensemble/model_*_weights.pth  (train_ensemble.py 输出)
  2. models/ensemble/model_*.pth          (train_ensemble.py checkpoint)
  3. models/best_model.pth               (train_67class.py 输出)
  4. models/best_transformer.pth         (train_transformer.py 输出)

用法:
  python src/demo.py
  python src/demo.py --model_dir models/ --num_samples 10
"""

import torch
import numpy as np
import os
import random
import argparse
import glob
from model import create_model
from data_utils import NUM_CLASSES, CLASS_NAMES

DATA_DIR = 'data/processed/'
MODEL_DIR = 'models/'


def load_model(model_path, device):
    """加载单个模型，兼容 checkpoint dict 和纯 state_dict 两种格式。"""
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        model_name = state.get('model_name', 'Gesture1DCNN')
        weights = state['model_state_dict']
        val_acc = state.get('best_acc', state.get('best_val_acc', 0.0))
        print(f'  来源: {os.path.basename(model_path)} (val_acc={val_acc:.4f})')
    else:
        model_name = 'Gesture1DCNN'
        weights = state
        print(f'  来源: {os.path.basename(model_path)}')

    n_cls = weights['fc.weight'].shape[0]
    m = create_model(model_name=model_name, input_channels=6, num_classes=n_cls).to(device)
    m.load_state_dict(weights)
    m.eval()
    return m, model_name, n_cls


def _find_model_files(model_dir):
    """按优先级搜索模型文件，返回路径列表。"""
    # 1) train_ensemble.py 输出：纯权重文件（优先，加载最快）
    weight_files = sorted(glob.glob(os.path.join(model_dir, 'ensemble', 'model_*_weights.pth')))
    if weight_files:
        return weight_files

    # 2) train_ensemble.py 输出：完整 checkpoint
    checkpoint_files = sorted(glob.glob(os.path.join(model_dir, 'ensemble', 'model_[0-9]*.pth')))
    # 过滤掉 _weights 后缀的（已在上一步处理）
    checkpoint_files = [f for f in checkpoint_files if '_weights' not in f]
    if checkpoint_files:
        return checkpoint_files

    # 3) train_67class.py 输出
    best_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(best_path):
        return [best_path]

    # 4) train_transformer.py 输出
    transformer_path = os.path.join(model_dir, 'best_transformer.pth')
    if os.path.exists(transformer_path):
        return [transformer_path]

    # 5) 旧版 train.py 遗留（向后兼容）
    legacy = sorted(glob.glob(os.path.join(model_dir, 'model_seed*.pth')))
    if legacy:
        return legacy

    return []


def load_ensemble(model_dir=MODEL_DIR):
    """自动发现并加载模型（集成或单模型）。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_files = _find_model_files(model_dir)
    if not model_files:
        raise FileNotFoundError(
            f"未找到模型文件。请先运行 train_ensemble.py 或 train_67class.py。\n"
            f"  搜索路径: {model_dir}\n"
            f"  期望文件: ensemble/model_*_weights.pth 或 best_model.pth"
        )

    models = []
    for f in model_files:
        m, mname, _ = load_model(f, device)
        models.append(m)

    if len(models) > 1:
        print(f"已加载 {len(models)} 个集成模型 (soft voting)")
    else:
        print(f"已加载单模型 ({mname}, {models[0].fc.out_features}类)")
    return models, device


def predict(models, device, sample):
    """集成推理（单模型退化为直接预测）。"""
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    n_cls = models[0].fc.out_features
    with torch.no_grad():
        logits = torch.zeros(1, n_cls).to(device)
        for m in models:
            logits += m(x)
        logits /= len(models)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = torch.argmax(logits, dim=1).item()
    top5_idx = np.argsort(probs)[-5:][::-1]
    return pred_idx, probs, top5_idx


def main():
    parser = argparse.ArgumentParser(description='AI-FitnessTrainer 推理演示')
    parser.add_argument('--model_dir', default=MODEL_DIR)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='展示样本数 (默认: 5)')
    args = parser.parse_args()

    x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy'),
                     allow_pickle=True).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'),
                     allow_pickle=True).astype(np.int64).flatten()

    models, device = load_ensemble(args.model_dir)
    n_cls = models[0].fc.out_features
    cls_names = CLASS_NAMES if len(CLASS_NAMES) == n_cls else [f'Class_{i}' for i in range(n_cls)]

    print(f"Device: {device}  |  类别: {n_cls}  |  测试样本: {len(x_test)}")
    print("=" * 70)

    # 每类选1个样本
    random.seed(42)
    samples_per_class = {}
    indices = list(range(len(x_test)))
    random.shuffle(indices)
    for idx in indices:
        cls = int(y_test[idx])
        if cls not in samples_per_class:
            samples_per_class[cls] = idx
        if len(samples_per_class) >= args.num_samples:
            break

    total_correct = 0
    total_top3 = 0
    print(f"\n每类随机选取 {args.num_samples} 个样本演示:\n")
    for cls in sorted(samples_per_class.keys()):
        idx = samples_per_class[cls]
        pred_idx, probs, top5 = predict(models, device, x_test[idx])

        correct = pred_idx == cls
        top3_correct = cls in top5[:3]
        total_correct += 1 if correct else 0
        total_top3 += 1 if top3_correct else 0

        cls_name = cls_names[cls] if cls < len(cls_names) else f'Class_{cls}'
        pred_name = cls_names[pred_idx] if pred_idx < len(cls_names) else f'Class_{pred_idx}'

        marker = '✓' if correct else ('○' if top3_correct else '✗')
        print(f"[{marker}] True: {cls_name} (id={cls})")
        print(f"      Pred: {pred_name} (id={pred_idx})")

        if n_cls <= 10:
            print(f"      Probs: " + " | ".join(
                f"{cls_names[i]}: {probs[i]:.2f}" for i in range(n_cls)))
        else:
            print(f"      Top-5: " + " | ".join(
                f"{cls_names[i]}: {probs[i]:.2f}" for i in top5))
        print()

    print(f"Top-1: {total_correct}/{len(samples_per_class)}")
    print(f"Top-3: {total_top3}/{len(samples_per_class)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
