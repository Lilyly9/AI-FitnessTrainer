"""
Rep Counter — IMU sensor based exercise repetition counting engine.

Algorithm:
  1. Select best axis (highest variance) or compute composite signal
  2. Estimate rep period via autocorrelation for adaptive min_peak_distance
  3. Apply Butterworth low-pass filter (zero-phase via filtfilt)
  4. Detect peaks via scipy.signal.find_peaks
  5. Edge artifact rejection

Supports both batch (count_reps) and streaming (count_reps_streaming) modes.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, correlate

SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

COLUMN_RENAME = {
    'ax': 'acc_x', 'ay': 'acc_y', 'az': 'acc_z',
    'gx': 'gyro_x', 'gy': 'gyro_y', 'gz': 'gyro_z',
    'exercise_type': 'label',
}


class RepCounter:
    """Count exercise repetitions from continuous 6-channel IMU data.

    Key parameters (tunable via CLI):
      - filter_cutoff: low-pass cutoff in Hz (default 2.5)
      - min_peak_distance: minimum samples between peaks, 0 = auto from autocorrelation
      - prominence: peak prominence in std units (default 0.3)
      - use_composite: True = magnitude composite, False = best single axis
    """

    def __init__(self, filter_cutoff=2.0, fs=100.0, min_peak_distance=0,
                 prominence=0.2, filter_order=4, use_composite=False):
        self.filter_cutoff = filter_cutoff
        self.fs = fs
        self.min_peak_distance = min_peak_distance
        self.prominence = prominence
        self.filter_order = filter_order
        self.use_composite = use_composite
        self._b = None
        self._a = None
        # Streaming state
        self._buffer = []
        self._confirmed_peaks = 0

    def _design_butterworth(self):
        if self._b is None:
            nyquist = 0.5 * self.fs
            cutoff_norm = self.filter_cutoff / nyquist
            if cutoff_norm >= 1.0:
                raise ValueError(
                    f"Filter cutoff ({self.filter_cutoff} Hz) must be < Nyquist ({nyquist} Hz)")
            self._b, self._a = butter(self.filter_order, cutoff_norm, btype='low')
        return self._b, self._a

    def apply_filter(self, signal):
        """Zero-phase low-pass filter. Falls back to raw if signal too short."""
        b, a = self._design_butterworth()
        pad_len = 3 * self.filter_order
        if len(signal) <= pad_len:
            return signal.copy()
        return filtfilt(b, a, signal)

    @staticmethod
    def _extract_signal(sensor_data, use_composite=False):
        """Extract a 1D signal from 6-channel IMU data.

        If use_composite: weighted combination of acc_mag + gyro_mag.
        Otherwise: the single channel with highest variance (z-scored).
        """
        if use_composite or sensor_data.shape[1] < 6:
            acc_mag = np.sqrt(
                sensor_data[:, 0] ** 2 +
                sensor_data[:, 1] ** 2 +
                sensor_data[:, 2] ** 2
            )
            gyro_mag = np.sqrt(
                sensor_data[:, 3] ** 2 +
                sensor_data[:, 4] ** 2 +
                sensor_data[:, 5] ** 2
            )
            acc_norm = (acc_mag - np.mean(acc_mag)) / (np.std(acc_mag) + 1e-8)
            gyro_norm = (gyro_mag - np.mean(gyro_mag)) / (np.std(gyro_mag) + 1e-8)
            return acc_norm + 0.5 * gyro_norm
        else:
            # Select channel with highest variance among all 6
            stds = [np.std(sensor_data[:, i]) for i in range(sensor_data.shape[1])]
            best = int(np.argmax(stds))
            raw = sensor_data[:, best]
            return (raw - np.mean(raw)) / (np.std(raw) + 1e-8)

    @staticmethod
    def _estimate_period(signal, fs=100.0, min_period_s=0.3, max_period_s=5.0):
        """Estimate dominant repetition period from autocorrelation.

        Returns period in samples, or 0 if no clear periodicity detected.
        """
        n = len(signal)
        if n < 50:
            return 0

        s = signal - np.mean(signal)
        if np.std(s) < 1e-8:
            return 0

        corr = correlate(s, s, mode='full')
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-8)

        min_lag = int(min_period_s * fs)
        max_lag = int(max_period_s * fs)
        max_lag = min(max_lag, len(corr) - 1)

        if min_lag >= len(corr):
            return 0

        search_range = corr[min_lag:max_lag + 1]
        if len(search_range) == 0:
            return 0

        # Find first prominent peak in autocorrelation
        peaks, props = find_peaks(search_range, prominence=0.05, distance=int(0.2 * fs))
        if len(peaks) == 0:
            return 0

        # Pick the highest prominence peak
        best = peaks[np.argmax(props['prominences'])]
        period = best + min_lag
        return period

    def count_reps(self, sensor_data):
        """Count repetitions from a continuous sensor data segment.

        Args:
            sensor_data: (T, 6) raw IMU array or (T,) 1D array

        Returns:
            (rep_count, peak_indices, filtered_signal, raw_signal)
        """
        # Ensure 2D
        if sensor_data.ndim == 1:
            sensor_data = sensor_data.reshape(-1, 1)

        raw = self._extract_signal(sensor_data, self.use_composite)
        filtered = self.apply_filter(raw)

        if np.std(filtered) < 1e-6:
            return 0, np.array([], dtype=int), filtered, raw

        # Adaptive min_peak_distance
        if self.min_peak_distance <= 0:
            period = self._estimate_period(filtered, self.fs)
            dist = max(int(period * 0.6), 30) if period > 0 else 50
        else:
            dist = self.min_peak_distance

        prominence_val = self.prominence * np.std(filtered)
        peaks, _ = find_peaks(filtered, distance=dist, prominence=prominence_val)

        # Edge rejection
        n = len(filtered)
        edge = max(dist, 20)
        peaks = peaks[(peaks >= edge) & (peaks <= n - edge)]

        return len(peaks), peaks, filtered, raw

    def count_reps_streaming(self, data_generator):
        """Count repetitions from a streaming data generator.

        Args:
            data_generator: yields (chunk_data, None) and (None, True) at end.
                           chunk_data has shape (T_chunk, 6).

        Yields:
            dict with keys: rep_count, new_peaks, total_samples, peaks, filtered, raw
        """
        self._buffer = []
        self._confirmed_peaks = 0
        prev_peak_count = 0

        for chunk, is_end in data_generator:
            if is_end:
                # Flush — all peaks now confirmed
                if self._buffer:
                    all_data = np.concatenate(self._buffer, axis=0)
                    count, peaks, filtered, raw = self.count_reps(all_data)
                    self._confirmed_peaks = count
                    yield {
                        'rep_count': count,
                        'new_peaks': count - prev_peak_count,
                        'total_samples': len(all_data),
                        'peaks': peaks, 'filtered': filtered, 'raw': raw,
                        'confirmed_mask': np.ones(len(peaks), dtype=bool),
                    }
                return

            self._buffer.append(chunk)
            all_data = np.concatenate(self._buffer, axis=0)

            min_len = 150
            if len(all_data) < min_len:
                yield {
                    'rep_count': 0, 'new_peaks': 0,
                    'total_samples': len(all_data),
                    'peaks': np.array([], dtype=int),
                    'filtered': np.array([]), 'raw': np.array([]),
                    'confirmed_mask': np.array([], dtype=bool),
                }
                continue

            count, peaks, filtered, raw = self.count_reps(all_data)

            # Only confirm peaks far enough from trailing edge
            n = len(filtered)
            if self.min_peak_distance <= 0:
                period = self._estimate_period(filtered, self.fs)
                edge_dist = max(int(period * 0.6), 30) if period > 0 else 50
            else:
                edge_dist = self.min_peak_distance

            confirmed = peaks <= (n - max(edge_dist, 20))
            self._confirmed_peaks = int(confirmed.sum())
            new_peaks = self._confirmed_peaks - prev_peak_count
            prev_peak_count = self._confirmed_peaks

            yield {
                'rep_count': int(self._confirmed_peaks),
                'new_peaks': new_peaks,
                'total_samples': len(all_data),
                'peaks': peaks,
                'filtered': filtered,
                'raw': raw,
                'confirmed_mask': confirmed,
            }


def read_raw_csv(filepath):
    """Read raw IMU CSV, standardize column names, extract sensor data.

    Returns:
        (sensor_data, labels, athlete_ids)
        sensor_data: (T, 6) float64 array
        labels: (T,) string array or None
        athlete_ids: (T,) int64 array or None
    """
    df = pd.read_csv(filepath)
    df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns},
              inplace=True)

    available_cols = [c for c in SENSOR_COLS if c in df.columns]
    df[available_cols] = df[available_cols].ffill()

    sensor_data = df[available_cols].values.astype(np.float64)
    labels = df['label'].values if 'label' in df.columns else None
    athlete_ids = df['athlete_id'].values if 'athlete_id' in df.columns else None

    return sensor_data, labels, athlete_ids


def chunk_generator(sensor_data, chunk_size=50, speed=1.0, fs=100.0,
                    start_idx=0, end_idx=None):
    """Generate chunks from sensor data, simulating real-time streaming.

    Yields:
        (chunk_data, None) for each chunk, (None, True) at end
    """
    import time
    if end_idx is None:
        end_idx = len(sensor_data)
    for i in range(start_idx, end_idx, chunk_size):
        chunk = sensor_data[i:i + chunk_size]
        yield (chunk, None)
        if speed > 0:
            time.sleep(len(chunk) / fs / speed)
    yield (None, True)
