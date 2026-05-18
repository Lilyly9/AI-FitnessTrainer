"""数据预处理单元测试。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import unittest
from datasets.preprocess_gym_gesture import create_windows


class TestCreateWindows(unittest.TestCase):
    def test_basic_shape(self):
        """通过构造带标签列的 DataFrame 测试窗口创建。"""
        data = np.random.randn(500, 6).astype(np.float32)
        labels = np.random.randint(0, 3, 500)
        # 构造含有 sensor 列和 label 列的 DataFrame
        cols = {f'col_{i}': data[:, i] for i in range(6)}
        cols['label'] = labels
        df = pd.DataFrame(cols)
        windows, win_labels = create_windows(
            df, window_size=200, step_size=100,
            sensor_cols=[f'col_{i}' for i in range(6)], label_col='label'
        )
        self.assertGreater(len(windows), 0)
        self.assertEqual(windows.shape[1:], (6, 200))


class TestDataFrameProcessing(unittest.TestCase):
    """通过 Gym Gesture 预处理脚本测试数据处理功能。"""
    def test_label_mapping(self):
        from datasets.preprocess_gym_gesture import LABEL_MAP
        self.assertEqual(LABEL_MAP['chest_fly'], 0)
        self.assertEqual(LABEL_MAP['chest_press'], 1)

    def test_sensor_cols_defined(self):
        from datasets.preprocess_gym_gesture import SENSOR_COLS
        self.assertEqual(len(SENSOR_COLS), 6)
        self.assertIn('acc_x', SENSOR_COLS)
        self.assertIn('gyro_z', SENSOR_COLS)


if __name__ == '__main__':
    unittest.main()
