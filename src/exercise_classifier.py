"""
Exercise Classifier — wrapper around the trained Gesture1DCNN model.

Provides clean interfaces for single-window, segment, and streaming classification.
Reuses model architecture from src/model.py and weight-loading pattern from src/demo.py.
"""

import torch
import numpy as np
import os
from collections import Counter, deque

try:
    from model import Gesture1DCNN
except ImportError:
    from src.model import Gesture1DCNN

CLASS_NAMES = ['chest_fly', 'chest_press', 'lat_pulldown',
               'seated_row', 'tricep_extension']
NUM_CLASSES = 5
WINDOW_SIZE = 200
STRIDE = 100


class ExerciseClassifier:
    """Wrapper around the trained Gesture1DCNN for exercise classification."""

    def __init__(self, model_dir='models/', num_classes=NUM_CLASSES,
                 class_names=None, device=None, use_ensemble=True):
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.use_ensemble = use_ensemble
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self._stream_buffer = None
        self._stream_preds = deque(maxlen=5)
        self._current_exercise = None

    def load_models(self):
        """Load trained model weights. Uses ensemble if available, else best_model.pth."""
        self.models = []
        if self.use_ensemble:
            seed_files = sorted([
                f for f in os.listdir(self.model_dir)
                if f.startswith('model_seed') and f.endswith('.pth')
            ])
        else:
            seed_files = []

        if seed_files:
            for f in seed_files:
                m = Gesture1DCNN(input_channels=6,
                                 num_classes=self.num_classes).to(self.device)
                state = torch.load(os.path.join(self.model_dir, f),
                                   map_location=self.device, weights_only=True)
                m.load_state_dict(state)
                m.eval()
                self.models.append(m)
            print(f"Loaded {len(self.models)} ensemble models: {seed_files}")
        else:
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            if os.path.exists(best_path):
                m = Gesture1DCNN(input_channels=6,
                                 num_classes=self.num_classes).to(self.device)
                state = torch.load(best_path, map_location=self.device,
                                   weights_only=True)
                m.load_state_dict(state)
                m.eval()
                self.models.append(m)
                print(f"Loaded single model: best_model.pth ({self.num_classes} classes)")
            else:
                raise FileNotFoundError(
                    f"No model files found in {self.model_dir}")

        return self

    def classify_window(self, window):
        """Classify a single window of IMU data.

        Args:
            window: (6, 200) numpy array (already Min-Max normalized)

        Returns:
            (class_name, class_id, probabilities)
        """
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = torch.zeros(1, self.num_classes).to(self.device)
            for m in self.models:
                logits += m(x)
            logits /= len(self.models)
            prob = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.class_names[pred_idx], pred_idx, prob.cpu().numpy()[0]

    def classify_segment(self, sensor_data):
        """Classify a continuous sensor data segment using sliding windows.

        Args:
            sensor_data: (T, 6) normalized IMU array

        Returns:
            dict with keys:
                class_name: majority-voted exercise name
                class_id: class index
                confidence: fraction of windows voting for majority class
                vote_distribution: dict {class_name: count}
                window_predictions: list of per-window class names
                num_windows: total windows processed
        """
        if len(sensor_data) < WINDOW_SIZE:
            raise ValueError(
                f"Segment too short ({len(sensor_data)} samples), need >= {WINDOW_SIZE}")

        all_preds = []
        for start in range(0, len(sensor_data) - WINDOW_SIZE + 1, STRIDE):
            window = sensor_data[start:start + WINDOW_SIZE].T  # -> (6, 200)
            class_name, _, _ = self.classify_window(window)
            all_preds.append(class_name)

        if not all_preds:
            raise ValueError("No valid windows extracted from segment")

        counter = Counter(all_preds)
        majority_class = counter.most_common(1)[0]
        class_name = majority_class[0]
        class_id = self.class_names.index(class_name)
        confidence = majority_class[1] / len(all_preds)

        return {
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence,
            'vote_distribution': dict(counter),
            'window_predictions': all_preds,
            'num_windows': len(all_preds),
        }

    def classify_stream(self, data_generator):
        """Streaming exercise classification with EMA smoothing.

        Args:
            data_generator: yields (chunk, None) for each normalized chunk,
                           (None, True) to signal end-of-stream.

        Yields:
            dict with exercise prediction after each chunk that yields valid windows
        """
        self._stream_buffer = None
        self._stream_preds.clear()
        self._current_exercise = None
        ema_votes = np.zeros(self.num_classes)

        for chunk, is_end in data_generator:
            if is_end:
                final = {'class_name': self._current_exercise or 'unknown',
                         'class_id': self.class_names.index(self._current_exercise)
                         if self._current_exercise in self.class_names else -1,
                         'confidence': 1.0 if self._current_exercise else 0.0,
                         'final': True}
                self._stream_buffer = None
                self._stream_preds.clear()
                yield final
                return

            if self._stream_buffer is None:
                self._stream_buffer = chunk.copy()
            else:
                self._stream_buffer = np.concatenate(
                    [self._stream_buffer, chunk], axis=0)

            # Slide windows through the data
            data = self._stream_buffer
            for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                window = data[start:start + WINDOW_SIZE].T
                class_name, class_id, prob = self.classify_window(window)
                self._stream_preds.append(class_id)
                # EMA update
                alpha = 0.3
                votes = np.zeros(self.num_classes)
                votes[class_id] = 1.0
                ema_votes = alpha * votes + (1 - alpha) * ema_votes

            # Determine current exercise from EMA
            if ema_votes.max() > 0.1:
                current_id = int(np.argmax(ema_votes))
                new_exercise = self.class_names[current_id]
                if new_exercise != self._current_exercise and len(self._stream_preds) >= 3:
                    # Only switch after 3 consecutive consistent predictions
                    recent = list(self._stream_preds)[-3:]
                    if len(set(recent)) == 1 and recent[0] == current_id:
                        self._current_exercise = new_exercise
                elif self._current_exercise is None:
                    self._current_exercise = new_exercise

            n_windows = (len(data) - WINDOW_SIZE) // STRIDE + 1
            yield {
                'class_name': self._current_exercise or 'unknown',
                'class_id': self.class_names.index(self._current_exercise)
                if self._current_exercise in self.class_names else -1,
                'confidence': float(ema_votes.max()) if ema_votes.max() > 0.1 else 0.0,
                'num_windows': max(n_windows, 0),
            }


def normalize_sensor_data(sensor_data, min_vals=None, max_vals=None):
    """Apply Min-Max normalization to sensor data.

    Args:
        sensor_data: (T, 6) array
        min_vals: (6,) precomputed min per channel, or None to compute from data
        max_vals: (6,) precomputed max per channel, or None to compute from data

    Returns:
        (normalized_data, min_vals, max_vals)
    """
    if min_vals is None:
        min_vals = sensor_data.min(axis=0)
    if max_vals is None:
        max_vals = sensor_data.max(axis=0)
    denom = max_vals - min_vals
    denom[denom < 1e-8] = 1.0
    normalized = (sensor_data - min_vals) / denom
    return normalized, min_vals, max_vals
