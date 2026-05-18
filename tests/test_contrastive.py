"""对比学习模块测试。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
import unittest
from contrastive import (
    ContrastiveAugment, ContrastiveDataset,
    SimCLRModel, ProjectionHead, nt_xent_loss,
)
from model import Gesture1DCNN


class TestContrastiveAugment(unittest.TestCase):
    def setUp(self):
        self.aug = ContrastiveAugment()

    def test_shape_preserved(self):
        x = np.random.randn(6, 200).astype(np.float32)
        for _ in range(20):
            x_aug = self.aug(x)
            self.assertEqual(x_aug.shape, (6, 200))

    def test_data_changed(self):
        aug = ContrastiveAugment(noise_std=0.1)
        x = np.ones((6, 200), dtype=np.float32)
        x_aug = aug(x)
        self.assertFalse(np.allclose(x, x_aug, atol=1e-4))

    def test_no_nan(self):
        x = np.random.randn(6, 200).astype(np.float32)
        for _ in range(50):
            x_aug = self.aug(x)
            self.assertFalse(np.isnan(x_aug).any())

    def test_two_views_different(self):
        x = np.random.randn(6, 200).astype(np.float32)
        v1 = self.aug(x)
        v2 = self.aug(x)
        self.assertFalse(np.allclose(v1, v2, atol=1e-6))


class TestNTXentLoss(unittest.TestCase):
    def test_positive_value(self):
        z_i = torch.randn(16, 64)
        z_j = torch.randn(16, 64)
        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        self.assertGreater(loss.item(), 0)

    def test_batch_size_1_no_crash(self):
        """batch_size=1 时无负样本，loss 为 0 是正常的（不应崩溃）。"""
        z_i = torch.randn(1, 64)
        z_j = torch.randn(1, 64)
        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)
        loss = nt_xent_loss(z_i, z_j)
        self.assertIsInstance(loss.item(), float)  # 不应崩溃


class TestProjectionHead(unittest.TestCase):
    def test_output_shape(self):
        head = ProjectionHead(input_dim=256, hidden_dim=128, output_dim=64)
        x = torch.randn(4, 256)
        out = head(x)
        self.assertEqual(out.shape, (4, 64))


class TestContrastiveDataset(unittest.TestCase):
    def test_returns_pair(self):
        data = np.random.randn(50, 6, 200).astype(np.float32)
        ds = ContrastiveDataset(data)
        xi, xj = ds[0]
        self.assertEqual(xi.shape, (6, 200))
        self.assertEqual(xj.shape, (6, 200))

    def test_two_views_different(self):
        data = np.random.randn(50, 6, 200).astype(np.float32)
        ds = ContrastiveDataset(data)
        xi, xj = ds[0]
        self.assertFalse(torch.allclose(xi, xj, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
