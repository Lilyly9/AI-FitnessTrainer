"""模型架构测试。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import torch.nn as nn
import unittest
from model import Gesture1DCNN


class TestGesture1DCNN(unittest.TestCase):
    def test_output_shape(self):
        model = Gesture1DCNN(input_channels=6, num_classes=5)
        x = torch.randn(4, 6, 200)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_different_input_channels(self):
        model = Gesture1DCNN(input_channels=3, num_classes=7)
        x = torch.randn(2, 3, 200)
        out = model(x)
        self.assertEqual(out.shape, (2, 7))

    def test_different_num_classes(self):
        model = Gesture1DCNN(input_channels=6, num_classes=25)
        x = torch.randn(8, 6, 200)
        out = model(x)
        self.assertEqual(out.shape, (8, 25))

    def test_parameter_count(self):
        model = Gesture1DCNN()
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(total, 40000)
        self.assertLess(total, 50000)

    def test_inference_mode(self):
        model = Gesture1DCNN().eval()
        with torch.no_grad():
            x = torch.randn(16, 6, 200)
            out = model(x)
            self.assertEqual(out.shape, (16, 5))
            self.assertFalse(torch.isnan(out).any())

    def test_gradient_flow(self):
        model = Gesture1DCNN()
        x = torch.randn(4, 6, 200)
        y = torch.randint(0, 5, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no gradient")


if __name__ == '__main__':
    unittest.main()
