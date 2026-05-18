import torch
from torch.utils.data import Dataset
import numpy as np


class GestureDataset(Dataset):
    """健身动作IMU数据集封装。训练模式开启数据增强。

    支持从文件路径或内存数组创建:
      GestureDataset('x.npy', 'y.npy', train=True)
      GestureDataset.from_arrays(x_arr, y_arr, train=True)
    """

    def __init__(self, x_path, y_path, train=True):
        self.x = np.load(x_path, allow_pickle=True).astype(np.float32)
        self.y = np.load(y_path, allow_pickle=True).astype(np.int64).flatten()
        self.train = train

    @classmethod
    def from_arrays(cls, x_array, y_array, train=True):
        """从内存数组创建数据集。"""
        ds = cls.__new__(cls)
        ds.x = np.asarray(x_array, dtype=np.float32)
        ds.y = np.asarray(y_array, dtype=np.int64).flatten()
        ds.train = train
        return ds

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        y = self.y[idx]

        if self.train:
            x = self._augment(x)

        return torch.tensor(x), torch.tensor(int(y), dtype=torch.long)

    @staticmethod
    def _augment(x):
        # 高斯噪声
        noise = np.random.randn(*x.shape).astype(np.float32) * 0.03
        x = x + noise
        # 随机幅度缩放
        scale = np.random.uniform(0.8, 1.2, size=(x.shape[0], 1)).astype(np.float32)
        x = x * scale
        # 随机时间偏移
        shift = np.random.randint(-15, 16)
        if shift != 0:
            x = np.roll(x, shift, axis=1)
            if shift > 0:
                x[:, :shift] = 0
            else:
                x[:, shift:] = 0
        # 时间扭曲
        t = np.arange(x.shape[1], dtype=np.float32)
        warp = np.random.uniform(0.9, 1.1)
        new_t = t * warp
        for c in range(x.shape[0]):
            x[c] = np.interp(t, new_t, x[c])
        return x
