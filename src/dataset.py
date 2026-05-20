"""
IMU 数据集封装 — 支持文件路径/内存数组两种创建方式。

增强的数据增强策略（6 种）:
  1) Jitter:      高斯噪声，模拟传感器噪声
  2) Scaling:     随机幅度缩放，模拟不同用力程度
  3) Time Warp:   时间轴扭曲，模拟不同动作速度
  4) Time Mask:   随机遮挡连续时间步，强制模型利用全局上下文
  5) Channel Drop: 随机丢弃一个通道（用该通道均值填充），模拟传感器故障
  6) Rotation:    加速度计通道随机旋转，模拟传感器佩戴角度变化

用法:
  from dataset import GestureDataset
  ds = GestureDataset('x_train.npy', 'y_train.npy', train=True,
                       noise_std=0.08, mask_ratio=0.1)
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
from collections import Counter


class GestureDataset(Dataset):
    """健身动作 IMU 数据集封装。训练模式开启增强数据增强。

    支持从文件路径或内存数组创建:
      GestureDataset('x.npy', 'y.npy', train=True)
      GestureDataset.from_arrays(x_arr, y_arr, train=True)

    可配置增强强度:
      ds = GestureDataset(..., noise_std=0.08, mask_ratio=0.1, rotation_prob=0.3)
    """

    def __init__(self, x_path, y_path, train=True,
                 noise_std=0.08, scale_range=(0.7, 1.3),
                 warp_range=(0.8, 1.2), shift_max=20,
                 mask_ratio=0.1, channel_drop_prob=0.15,
                 rotation_prob=0.3):
        self.x = np.load(x_path, allow_pickle=True).astype(np.float32)
        self.y = np.load(y_path, allow_pickle=True).astype(np.int64).flatten()
        self.train = train

        # 增强参数
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.warp_range = warp_range
        self.shift_max = shift_max
        self.mask_ratio = mask_ratio
        self.channel_drop_prob = channel_drop_prob
        self.rotation_prob = rotation_prob

    @classmethod
    def from_arrays(cls, x_array, y_array, train=True, **kwargs):
        """从内存数组创建数据集。"""
        ds = cls.__new__(cls)
        ds.x = np.asarray(x_array, dtype=np.float32)
        ds.y = np.asarray(y_array, dtype=np.int64).flatten()
        ds.train = train
        ds.noise_std = kwargs.get('noise_std', 0.08)
        ds.scale_range = kwargs.get('scale_range', (0.7, 1.3))
        ds.warp_range = kwargs.get('warp_range', (0.8, 1.2))
        ds.shift_max = kwargs.get('shift_max', 20)
        ds.mask_ratio = kwargs.get('mask_ratio', 0.1)
        ds.channel_drop_prob = kwargs.get('channel_drop_prob', 0.15)
        ds.rotation_prob = kwargs.get('rotation_prob', 0.3)
        return ds

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        y = self.y[idx]

        if self.train:
            x = self._augment(x)

        return torch.tensor(x), torch.tensor(int(y), dtype=torch.long)

    def _augment(self, x):
        """增强增强 pipeline — 按概率独立应用多种变换。"""
        # 1) Jitter: 高斯噪声（强度可调）
        if np.random.rand() < 0.9:
            noise = np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            x = x + noise

        # 2) Scaling: 幅度缩放（模拟不同用力程度）
        if np.random.rand() < 0.6:
            scale = np.random.uniform(*self.scale_range, size=(x.shape[0], 1)).astype(np.float32)
            x = x * scale

        # 3) Time Warp: 时间轴扭曲（模拟不同动作速度）
        if np.random.rand() < 0.5:
            t = np.arange(x.shape[1], dtype=np.float32)
            warp = np.random.uniform(*self.warp_range)
            for c in range(x.shape[0]):
                x[c] = np.interp(t, t * warp, x[c])

        # 4) Time Mask: 随机遮挡连续时间步
        if np.random.rand() < 0.5:
            T = x.shape[1]
            mask_len = max(4, int(T * np.random.uniform(0.05, self.mask_ratio)))
            if mask_len < T - 1:
                start = np.random.randint(0, T - mask_len)
                # 用该通道均值填充而非置零（更温和）
                for c in range(x.shape[0]):
                    ch_mean = x[c].mean()
                    x[c, start:start + mask_len] = ch_mean

        # 5) Channel Drop: 随机丢弃一个通道
        if np.random.rand() < self.channel_drop_prob:
            drop_ch = np.random.randint(0, x.shape[0])
            x[drop_ch] = x[drop_ch].mean()  # 用均值填充而非置零

        # 6) Rotation: 加速度计三轴随机旋转（模拟传感器佩戴角度变化）
        if np.random.rand() < self.rotation_prob and x.shape[0] >= 3:
            x = self._rotate_acc(x)

        # 7) Time Shift: 时间偏移
        if np.random.rand() < 0.3:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            if shift != 0:
                x = np.roll(x, shift, axis=1)
                if shift > 0:
                    x[:, :shift] = 0
                else:
                    x[:, shift:] = 0

        return x

    def _rotate_acc(self, x):
        """对加速度计三轴 (ch0-ch2) 施加随机 3D 旋转。

        模拟传感器在手腕上的不同佩戴角度，让模型学到方向不变性。
        陀螺仪通道 (ch3-ch5) 做对应旋转变换。
        """
        # 随机旋转角度（限制在 ±30° 内，避免完全颠倒）
        angle_x = np.random.uniform(-np.pi / 6, np.pi / 6)
        angle_y = np.random.uniform(-np.pi / 6, np.pi / 6)
        angle_z = np.random.uniform(-np.pi / 6, np.pi / 6)

        # 构建旋转矩阵 Rx * Ry * Rz
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx  # (3, 3)

        # 旋转加速度计 (ch0-ch2)
        acc = x[:3, :]  # (3, T)
        acc_rotated = R @ acc  # (3, T)
        x[:3, :] = acc_rotated

        # 旋转陀螺仪 (ch3-ch5) — 角速度同样受旋转影响
        if x.shape[0] >= 6:
            gyr = x[3:6, :]  # (3, T)
            gyr_rotated = R @ gyr
            x[3:6, :] = gyr_rotated

        return x


def create_balanced_sampler(y, num_samples=None):
    """创建加权采样器，使每个 batch 内类别均衡。

    参数:
      y: 训练标签数组 (N,) 或 标签文件路径
      num_samples: 每个 epoch 的总采样数（默认 = len(y)，即一轮过完所有类）

    返回:
      WeightedRandomSampler

    用法:
      sampler = create_balanced_sampler('data/processed/y_train.npy')
      loader = DataLoader(ds, batch_size=64, sampler=sampler)
    """
    if isinstance(y, str):
        y = np.load(y).astype(np.int64).flatten()
    else:
        y = np.asarray(y, dtype=np.int64).flatten()

    counter = Counter(y)
    n_samples = len(y)
    n_classes = len(counter)

    # 反频率权重：样本越少的类，被采样概率越高
    class_weights = np.ones(n_classes, dtype=np.float32)
    for cls, count in counter.items():
        class_weights[cls] = n_samples / (n_classes * count)

    sample_weights = class_weights[y]

    if num_samples is None:
        num_samples = n_samples

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=num_samples,
        replacement=True,
    )
