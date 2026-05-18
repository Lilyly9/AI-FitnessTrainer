"""运动分析模块测试。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import unittest
from analysis import (
    compute_signal_energy, score_exercise_quality,
    segment_into_sets, detect_anomaly, full_analysis,
)


def make_simulated_exercise(n_cycles=10, cycle_len=100, noise_level=0.02):
    t = np.linspace(0, 2 * np.pi * n_cycles, n_cycles * cycle_len)
    acc_x = 0.5 * np.sin(t) + noise_level * np.random.randn(len(t))
    acc_y = 0.3 * np.sin(t + 0.5) + noise_level * np.random.randn(len(t))
    acc_z = 0.7 * np.sin(t - 0.3) + noise_level * np.random.randn(len(t))
    gyro_x = 0.2 * np.cos(t) + noise_level * np.random.randn(len(t)) * 0.5
    gyro_y = 0.15 * np.cos(t + 0.4) + noise_level * np.random.randn(len(t)) * 0.5
    gyro_z = 0.1 * np.cos(t - 0.2) + noise_level * np.random.randn(len(t)) * 0.5
    return np.column_stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]).astype(np.float32)


class TestComputeSignalEnergy(unittest.TestCase):
    def test_shape(self):
        data = np.random.randn(500, 6).astype(np.float32)
        energy = compute_signal_energy(data)
        self.assertEqual(energy.shape, (500,))

    def test_non_negative(self):
        data = np.random.randn(500, 6).astype(np.float32)
        energy = compute_signal_energy(data)
        self.assertTrue((energy >= 0).all())


class TestSegmentIntoSets(unittest.TestCase):
    def test_single_set(self):
        pos = [100, 300, 500, 700, 900]
        sets = segment_into_sets(pos, max_rest_samples=2000)
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0]['reps'], 5)

    def test_multiple_sets(self):
        pos = [100, 300, 500, 5000, 5200, 5400]
        sets = segment_into_sets(pos, max_rest_samples=2000)
        self.assertEqual(len(sets), 2)


class TestScoreQuality(unittest.TestCase):
    def test_returns_all_keys(self):
        data = make_simulated_exercise(n_cycles=10, cycle_len=100)
        result = score_exercise_quality(data)
        for key in ['overall_score', 'consistency_score', 'smoothness_score',
                     'range_of_motion_score', 'feedback']:
            self.assertIn(key, result)

    def test_score_in_range(self):
        data = make_simulated_exercise(n_cycles=10, cycle_len=100)
        result = score_exercise_quality(data)
        for key in ['overall_score', 'consistency_score', 'smoothness_score',
                     'range_of_motion_score']:
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 100)


class TestDetectAnomaly(unittest.TestCase):
    def test_no_crash(self):
        data = make_simulated_exercise(n_cycles=10, cycle_len=100)
        anomalies = detect_anomaly(data)
        self.assertIsInstance(anomalies, list)


class TestFullAnalysis(unittest.TestCase):
    def test_returns_all_sections(self):
        data = make_simulated_exercise(n_cycles=10, cycle_len=100)
        result = full_analysis(data)
        for key in ['exercise', 'sets', 'quality', 'anomalies']:
            self.assertIn(key, result)


if __name__ == '__main__':
    unittest.main()
