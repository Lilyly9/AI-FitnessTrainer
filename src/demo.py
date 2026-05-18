import torch
import numpy as np
import os
import random
from model import Gesture1DCNN
from data_utils import NUM_CLASSES, CLASS_NAMES

DATA_DIR = 'data/processed/'
MODEL_DIR = 'models/'


def load_ensemble():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    seed_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_seed') and f.endswith('.pth')]
    if seed_files:
        for f in sorted(seed_files):
            m = Gesture1DCNN(input_channels=6, num_classes=NUM_CLASSES).to(device)
            m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f), map_location=device))
            m.eval()
            models.append(m)
        print(f"Loaded {len(models)} ensemble models: {seed_files}")
    else:
        m = Gesture1DCNN(input_channels=6, num_classes=NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), map_location=device))
        m.eval()
        models.append(m)
        print("Loaded single model: best_model.pth")
    return models, device


def predict(models, device, sample):
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = torch.zeros(1, NUM_CLASSES).to(device)
        for m in models:
            logits += m(x)
        logits /= len(models)
        prob = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
    return CLASS_NAMES[pred_idx], pred_idx, prob.cpu().numpy()[0]


def main():
    x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy')).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy')).astype(np.int64)

    models, device = load_ensemble()
    print(f"Device: {device}")
    print(f"Test samples: {len(x_test)} windows")
    print("=" * 60)

    # 每类随机选1个共5个样本展示
    random.seed(42)
    samples_per_class = {}
    indices = list(range(len(x_test)))
    random.shuffle(indices)
    for idx in indices:
        cls = y_test[idx]
        if cls not in samples_per_class:
            samples_per_class[cls] = idx
        if len(samples_per_class) == 5:
            break

    print("\nOne sample per class:\n")
    correct = 0
    for cls in sorted(samples_per_class.keys()):
        idx = samples_per_class[cls]
        sample = x_test[idx]
        true_label = y_test[idx]
        pred_name, pred_idx, probs = predict(models, device, sample)

        is_correct = pred_idx == true_label
        if is_correct:
            correct += 1

        print(f"[{'OK' if is_correct else 'X'}] True: {CLASS_NAMES[true_label]:<18s} -> Pred: {pred_name}")
        print(f"    Probabilities: " +
              " | ".join(f"{name}: {p:.2f}" for name, p in zip(CLASS_NAMES, probs)))
        print()

    print(f"Result: {correct}/5 correct")
    print("=" * 60)


if __name__ == '__main__':
    main()
