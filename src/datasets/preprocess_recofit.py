"""
Microsoft RecoFit 数据集预处理 — Domain Adaptation 版本
================================================================
目标：将 forearm (RecoFit) 数据转换为更接近 wrist (GymGesture/MM-Fit) 的分布，
      缩小 domain gap，使模型能在 wrist 数据上更好地泛化。

核心改进：
  1. PCA 方向归一化：将 YZ 平面信号转换为方向不变特征
     (acc_y, acc_z) → PCA → (acc_pc1, acc_mag)
     (gyr_y, gyr_z) → PCA → (gyr_pc1, gyr_mag)
     输出 6 通道: [acc_x, acc_pc1, acc_mag, gyr_x, gyr_pc1, gyr_mag]

  2. Z-score 归一化：替代 Min-Max，保留通道间相对关系

  3. 类别过滤：仅保留与 wrist 数据集重叠的 10 个动作

  4. Wrist-style 数据增强：YZ 平面随机旋转、传感器漂移、重力漂移

使用方法:
  1. 下载 exercise_data.50.0000_multionly.mat 放到 data/raw/recofit/
  2. python src/datasets/preprocess_recofit.py
  3. python src/datasets/preprocess_recofit.py --augment 2  # 每个样本生成2个增强副本

输出: data/processed/recofit_x_train.npy, recofit_y_train.npy, ...
      以及 recofit_label_mapping.csv, recofit_norm_params.npz
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
import time
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
DATA_DIR = 'data/raw/recofit/'
OUT_DIR = 'data/processed/'
MAT_FILENAME = 'exercise_data.50.0000_multionly.mat'
WINDOW_SIZE = 200
STEP_SIZE = 20          # 步长 20，确保每类样本数充足
TEST_RATIO = 0.2
RANDOM_SEED = 42

# RecoFit 传感器佩戴位置：前臂 (arm-band)
DOMAIN_ID = 0           # 0 = forearm
SENSOR_POSITION = 'forearm'

# ==================== 跳过的不相关活动 ====================
SKIP_ACTIVITIES = {
    '<Initial Activity>', 'Tap Left Device', 'Tap Right Device',
    'Non-Exercise', 'Device on Table', 'Arm Band Adjustment',
}

# ==================== RecoFit 原始名称 → 目标 canonical 名称 ====================
# 仅保留与 wrist 数据集 (GymGesture + MM-Fit) 重叠的 10 个动作类别
RECOFIT_TO_TARGET = {
    # ---- bicep_curl ----
    'Bicep Curl':                    'bicep_curl',
    'Two-arm Dumbbell Curl':         'bicep_curl',
    'Alternating Dumbbell Curl':     'bicep_curl',
    'Biceps Curl (band)':            'bicep_curl',
    # ---- jumping_jack ----
    'Jumping Jacks':                 'jumping_jack',
    # ---- pushup ----
    'Pushups':                       'pushup',
    'Pushup (variation)':            'pushup',
    # ---- squat (含多种深蹲变体) ----
    'Squat':                         'squat',
    'Squat (arms front)':            'squat',
    'Squat Jump':                    'squat',
    'Squat (kettlebell)':            'squat',
    'Squat (hands behind)':          'squat',
    'Dumbbell Squat (hands side)':   'squat',
    'Wall Squat':                    'squat',
    # ---- lunge ----
    'Walking lunge':                 'lunge',
    'Lunge (alternating)':           'lunge',
    # ---- lateral_raise ----
    'Lateral Raise':                 'lateral_raise',
    # ---- tricep_extension ----
    'Overhead Triceps Ext':          'tricep_extension',
    'Triceps Kickback (both)':       'tricep_extension',
    'Triceps Kickback (right)':      'tricep_extension',
    'Triceps Kickback (left)':       'tricep_extension',
    'Triceps ext (lying)':           'tricep_extension',
    'Overhead Triceps Ext (both)':   'tricep_extension',
    # ---- dumbbell_row ----
    'Dumbbell Deadlift Row':         'dumbbell_row',
    'Dumbbell Row (both arms)':      'dumbbell_row',
    'Dumbbell Row (right)':          'dumbbell_row',
    'Dumbbell Row (left)':           'dumbbell_row',
    # ---- dumbbell_shoulder_press ----
    'Shoulder Press':                'dumbbell_shoulder_press',
    'Squat Rack Shoulder Press':     'dumbbell_shoulder_press',
    # ---- situp ----
    'Sit-ups':                       'situp',
    'Sit-up (hands behind)':         'situp',
}

# 目标 canonical 名称 → 本地标签 (0-9)
TARGET_CANONICAL_NAMES = [
    'bicep_curl',
    'dumbbell_row',
    'dumbbell_shoulder_press',
    'jumping_jack',
    'lateral_raise',
    'lunge',
    'pushup',
    'situp',
    'squat',
    'tricep_extension',
]
TARGET_LABEL_MAP = {name: i for i, name in enumerate(TARGET_CANONICAL_NAMES)}

# 统计每个 canonical 类别对应的 RecoFit 原始名称（用于报告）
CANONICAL_TO_RECOFIT_NAMES = {}
for reco_name, canon in RECOFIT_TO_TARGET.items():
    CANONICAL_TO_RECOFIT_NAMES.setdefault(canon, []).append(reco_name)


# ============================================================
# 辅助工具
# ============================================================

def _extract_scalar(value):
    """从嵌套 numpy array 中递归提取标量值。"""
    while isinstance(value, np.ndarray) and value.size > 0:
        value = value.ravel()[0]
    return value


def _extract_str(value):
    """从嵌套 numpy array 中递归提取字符串。"""
    while isinstance(value, np.ndarray) and value.size > 0:
        value = value.ravel()[0]
    return str(value)


# ============================================================
# 1. PCA 方向归一化
# ============================================================

def pca_orientation_normalize_single(window):
    """对单个窗口做 PCA 方向归一化，实现 YZ 平面旋转不变性。

    参数:
      window: (6, 200) — 原始 6 通道 IMU 窗口
              ch0=acc_x, ch1=acc_y, ch2=acc_z
              ch3=gyr_x, ch4=gyr_y, ch5=gyr_z

    返回:
      new_window: (6, 200) — PCA 变换后的 6 通道
                  ch0=acc_x (保留), ch1=acc_pc1, ch2=acc_mag
                  ch3=gyr_x (保留), ch4=gyr_pc1, ch5=gyr_mag

    原理:
      无论传感器绕 X 轴（前臂）旋转多少度，YZ 平面的主运动方向 (pc1)
      和运动幅度 (mag) 是不变的。这消除了 forearm 与 wrist 之间
      因佩戴角度不同导致的分布差异。
    """
    new_window = np.zeros_like(window, dtype=np.float32)

    # 保留 X 轴（沿手臂方向，不受 YZ 旋转影响）
    new_window[0] = window[0].copy()  # acc_x
    new_window[3] = window[3].copy()  # gyr_x

    # ---- 加速度计 YZ 平面 PCA ----
    acc_y = window[1]  # (200,)
    acc_z = window[2]  # (200,)
    acc_pc1, acc_mag = _yz_pca(acc_y, acc_z)
    new_window[1] = acc_pc1
    new_window[2] = acc_mag

    # ---- 陀螺仪 YZ 平面 PCA ----
    gyr_y = window[4]  # (200,)
    gyr_z = window[5]  # (200,)
    gyr_pc1, gyr_mag = _yz_pca(gyr_y, gyr_z)
    new_window[4] = gyr_pc1
    new_window[5] = gyr_mag

    return new_window


def _yz_pca(y, z):
    """对 YZ 平面的 2D 数据做 PCA，提取主成分和幅度。

    参数:
      y, z: (T,) — Y 和 Z 通道的时序数据

    返回:
      pc1: (T,) — 投影到第一主成分的值（去均值）
      mag: (T,) — sqrt(y² + z²) 运动幅度
    """
    T = len(y)
    yz = np.stack([y, z], axis=1)  # (T, 2)
    yz_centered = yz - yz.mean(axis=0, keepdims=True)

    # 协方差矩阵 (2, 2) 的闭式特征分解
    cov = np.cov(yz_centered.T)  # (2, 2)
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]

    # 特征值: λ = (a+c ± √((a-c)² + 4b²)) / 2
    trace = a + c
    det = a * c - b * b
    discriminant = max(trace * trace - 4 * det, 0)
    sqrt_disc = np.sqrt(discriminant)
    lambda_max = (trace + sqrt_disc) / 2.0

    # 最大特征值对应的特征向量
    if abs(b) > 1e-10:
        # v = [b, λ_max - a]，然后归一化
        v = np.array([b, lambda_max - a], dtype=np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm
        else:
            v = np.array([1.0, 0.0], dtype=np.float32)
    elif a >= c:
        v = np.array([1.0, 0.0], dtype=np.float32)
    else:
        v = np.array([0.0, 1.0], dtype=np.float32)

    # 投影到第一主成分
    pc1 = yz_centered @ v  # (T,)

    # 运动幅度（旋转不变量）
    mag = np.sqrt(y * y + z * z)

    return pc1.astype(np.float32), mag.astype(np.float32)


def batch_pca_orientation_normalize(X, verbose=True):
    """批量 PCA 方向归一化。

    参数:
      X: (N, 6, 200) — 原始 IMU 窗口
      verbose: 是否打印进度

    返回:
      X_new: (N, 6, 200) — PCA 变换后的窗口
    """
    N = X.shape[0]
    X_new = np.zeros_like(X, dtype=np.float32)

    for i in range(N):
        X_new[i] = pca_orientation_normalize_single(X[i])
        if verbose and (i + 1) % 5000 == 0:
            print(f"  PCA 进度: {i + 1}/{N}")

    return X_new


# ============================================================
# 2. Z-score 归一化
# ============================================================

def compute_zscore_params(X_train):
    """在训练集上计算 per-channel 均值和标准差。

    参数:
      X_train: (N, C, T)

    返回:
      mean: (C, 1, 1)
      std:  (C, 1, 1)
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)  # 防止除零
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X, mean, std):
    """应用 Z-score 归一化: (x - mean) / std。"""
    return (X.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)


# ============================================================
# 4. Wrist-style 数据增强
# ============================================================

def yz_plane_rotation(x):
    """YZ 平面随机旋转 — 模拟前臂旋转导致的传感器投影变化。

    对 PCA 空间中的 (acc_pc1, acc_mag) 和 (gyr_pc1, gyr_mag) 施加
    相同的随机 2D 旋转角度，模拟 wrist 传感器的旋转自由度。

    参数:
      x: (6, T) — PCA 变换后的窗口

    返回:
      x: (6, T) — 旋转后的窗口
    """
    theta = np.random.uniform(-np.pi / 4, np.pi / 4)  # ±45°
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    # 旋转加速度计 PCA 分量 (ch1=acc_pc1, ch2=acc_mag)
    acc_pca = x[1:3, :]  # (2, T)
    x[1:3, :] = R @ acc_pca

    # 旋转陀螺仪 PCA 分量 (ch4=gyr_pc1, ch5=gyr_mag)
    gyr_pca = x[4:6, :]  # (2, T)
    x[4:6, :] = R @ gyr_pca

    return x


def sensor_drift(x, max_drift=0.08):
    """传感器漂移模拟 — 添加缓慢线性基线漂移。

    模拟传感器温漂、电路偏移等长时间缓慢变化，wrist 传感器因更靠近
    皮肤表面温度变化更大，漂移效应更明显。

    参数:
      x: (6, T) — 窗口
      max_drift: 最大漂移幅度（相对于归一化后数据）

    返回:
      x: (6, T) — 添加漂移后的窗口
    """
    T = x.shape[1]
    # 每个通道独立的随机漂移方向和幅度
    drift_vals = np.random.uniform(-max_drift, max_drift, size=(x.shape[0], 1)).astype(np.float32)
    # 缓慢线性斜坡
    ramp = np.linspace(0, 1, T, dtype=np.float32).reshape(1, T)
    x = x + drift_vals * ramp
    return x


def gravity_drift(x, max_drift=0.12):
    """重力漂移模拟 — 在加速度计通道添加缓慢正弦波动。

    手腕在运动过程中会自然旋转，导致重力在加速度计各轴上的投影
    缓慢变化。这模拟了 wrist IMU 特有的重力分量漂移特征。

    参数:
      x: (6, T) — 窗口
      max_drift: 最大漂移幅度（相对于归一化后数据）

    返回:
      x: (6, T) — 添加重力漂移后的窗口
    """
    T = x.shape[1]
    # 随机频率 (0.3-1.5 个完整周期)、随机相位
    freq = np.random.uniform(0.3, 1.5)
    phase = np.random.uniform(0, 2 * np.pi)
    amplitude = np.random.uniform(0.02, max_drift)

    drift = amplitude * np.sin(2 * np.pi * freq * np.arange(T, dtype=np.float32) / T + phase)

    # 仅在加速度计相关通道 (ch0=acc_x, ch1=acc_pc1, ch2=acc_mag) 上添加
    x[0:3] += drift.reshape(1, T).astype(np.float32)

    return x


def time_warp_wrist(x, warp_range=(0.85, 1.15)):
    """时间扭曲 — 模拟 wrist 动作速度的自然变化。

    手腕动作比前臂动作更灵活，速度变化范围更大。

    参数:
      x: (6, T) — 窗口
      warp_range: 拉伸范围

    返回:
      x: (6, T) — 时间扭曲后的窗口
    """
    T = x.shape[1]
    t_orig = np.arange(T, dtype=np.float32)
    warp_factor = np.random.uniform(*warp_range)
    t_warped = t_orig * warp_factor

    for c in range(x.shape[0]):
        x[c] = np.interp(t_orig, t_warped, x[c]).astype(np.float32)

    return x


def wrist_augment(x, apply_prob=0.7):
    """综合 wrist-style 增强 pipeline。

    按概率独立应用多种 wrist 特征增强，让 forearm 数据分布
    更接近 wrist 数据。

    参数:
      x: (6, T) — PCA 变换 + Z-score 归一化后的窗口
      apply_prob: 每种增强的独立应用概率

    返回:
      x: (6, T) — 增强后的窗口
    """
    x = x.copy()

    # 1) YZ 平面随机旋转 — 核心 wrist 特征
    if np.random.rand() < apply_prob:
        x = yz_plane_rotation(x)

    # 2) 传感器漂移
    if np.random.rand() < apply_prob * 0.6:
        x = sensor_drift(x)

    # 3) 重力漂移
    if np.random.rand() < apply_prob * 0.5:
        x = gravity_drift(x)

    # 4) 时间扭曲 — wrist 动作速度变化更大
    if np.random.rand() < apply_prob * 0.5:
        x = time_warp_wrist(x)

    return x


def generate_augmented_samples(X_train, y_train, num_copies=2, verbose=True):
    """为训练集生成 wrist-style 增强样本。

    参数:
      X_train: (N, 6, 200) — 原始训练集
      y_train: (N,) — 训练标签
      num_copies: 每个原始样本生成的增强副本数
      verbose: 是否打印进度

    返回:
      X_aug: (N * num_copies, 6, 200) — 增强样本
      y_aug: (N * num_copies,) — 对应的标签
    """
    N = X_train.shape[0]
    X_aug_list = []
    y_aug_list = []

    for copy_idx in range(num_copies):
        if verbose:
            print(f"  生成增强副本 {copy_idx + 1}/{num_copies}...")
        X_copy = np.zeros_like(X_train, dtype=np.float32)
        for i in range(N):
            X_copy[i] = wrist_augment(X_train[i])
            if verbose and (i + 1) % 5000 == 0:
                print(f"    增强进度: {i + 1}/{N}")
        X_aug_list.append(X_copy)
        y_aug_list.append(y_train.copy())

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    return X_aug, y_aug


# ============================================================
# 数据提取
# ============================================================

def extract_segments(mat_data):
    """从 .mat 数据结构中提取活动片段（仅保留目标类别）。

    参数:
      mat_data: scipy.io.loadmat 加载的 MATLAB 数据

    返回:
      segments: list of dict, 每个元素包含 'acc', 'gyr', 'label', 'canonical', 'subject_id'
      subjects: (M,) — 每个片段的受试者 ID
    """
    subject_data = mat_data['subject_data']
    num_subjects = subject_data.shape[0]
    all_segments = []
    all_subjects = []

    matched_classes = set()
    total_skipped = 0

    for subj_idx in range(num_subjects):
        records = subject_data[subj_idx, 0]  # shape (N,)

        for rec_idx in range(records.shape[0]):
            record = records[rec_idx]
            act_mat_1d = record['activityStartMatrix']  # (K,)
            data_1d = record['data']                    # (K,)

            k = min(len(act_mat_1d), len(data_1d))
            if k == 0:
                continue

            # 受试者 ID
            subj_id = _extract_scalar(record['subjectID'])
            try:
                subj_id = int(subj_id)
            except (ValueError, TypeError):
                pass

            for seg_idx in range(k):
                act_block = act_mat_1d[seg_idx]         # (M, 7)
                data_struct = data_1d[seg_idx]

                acc_full = data_struct['accelDataMatrix'][0, 0]  # (T, 4)
                gyr_full = data_struct['gyroDataMatrix'][0, 0]   # (T, 4)

                if acc_full.ndim != 2 or acc_full.shape[1] < 4:
                    continue
                if gyr_full.ndim != 2 or gyr_full.shape[1] < 4:
                    continue

                # 遍历该片段内的每个小活动
                for row_idx in range(act_block.shape[0]):
                    act_name = _extract_str(act_block[row_idx, 0])

                    # 跳过不需要的活动
                    if act_name in SKIP_ACTIVITIES:
                        continue

                    # 仅保留目标类别
                    if act_name not in RECOFIT_TO_TARGET:
                        total_skipped += 1
                        continue

                    canonical = RECOFIT_TO_TARGET[act_name]
                    target_label = TARGET_LABEL_MAP[canonical]
                    matched_classes.add(act_name)

                    # 根据帧号切割
                    seq_info = act_block[row_idx, -1]
                    start_seq = _extract_scalar(seq_info['startSequenceNumberMaster'])
                    end_seq = _extract_scalar(seq_info['endSequenceNumberMaster'])

                    start_idx = int(start_seq) - 1
                    end_idx = int(end_seq) - 1

                    # 裁剪到有效范围
                    if start_idx < 0:
                        start_idx = 0
                    if end_idx >= acc_full.shape[0]:
                        end_idx = acc_full.shape[0] - 1
                    if end_idx <= start_idx:
                        continue

                    # 提取加速度计和陀螺仪数据 (去除时间戳列)
                    acc_seg = acc_full[start_idx:end_idx + 1, 1:].astype(np.float32)  # (seg_len, 3)
                    gyr_seg = gyr_full[start_idx:end_idx + 1, 1:].astype(np.float32)

                    all_segments.append({
                        'acc': acc_seg,          # (L, 3): [x, y, z]
                        'gyr': gyr_seg,          # (L, 3): [x, y, z]
                        'label': target_label,    # 目标类别标签 (0-9)
                        'canonical': canonical,   # canonical 名称
                        'recofit_name': act_name, # RecoFit 原始名称
                        'subject_id': subj_id,
                    })
                    all_subjects.append(subj_id)

    print(f"  目标类别匹配: {len(matched_classes)} 个 RecoFit 原始名称")
    for canon in TARGET_CANONICAL_NAMES:
        names = CANONICAL_TO_RECOFIT_NAMES.get(canon, [])
        matched = [n for n in names if n in matched_classes]
        if matched:
            print(f"    [{TARGET_LABEL_MAP[canon]}] {canon}: {len(matched)} 变体 {matched}")
        else:
            print(f"    [{TARGET_LABEL_MAP[canon]}] {canon}: ⚠ 无匹配数据!")

    if total_skipped > 0:
        print(f"  已跳过 {total_skipped} 个非目标活动块")

    return all_segments, np.array(all_subjects)


def sliding_window(segments, window_size, step_size):
    """滑动窗口切分。

    参数:
      segments: 活动片段列表
      window_size: 窗口大小 (时间步)
      step_size: 滑动步长

    返回:
      X: (N, 6, window_size) — 窗口数据 [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
      y: (N,) — 标签
      subjects: (N,) — 受试者 ID
    """
    X_list, y_list, subj_list = [], [], []

    for seg in segments:
        acc = seg['acc']  # (L, 3)
        gyr = seg['gyr']  # (L, 3)
        length = acc.shape[0]

        for start in range(0, length - window_size + 1, step_size):
            end = start + window_size
            # 拼接为 6 通道: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            win = np.concatenate([
                acc[start:end, :].T,
                gyr[start:end, :].T,
            ], axis=0).astype(np.float32)  # (6, window_size)
            X_list.append(win)
            y_list.append(seg['label'])
            subj_list.append(seg['subject_id'])

    X = np.stack(X_list, axis=0)  # (N, 6, window_size)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subj_list)
    return X, y, subjects


# ============================================================
# 主流程
# ============================================================

def preprocess(augment_copies=0):
    """RecoFit 数据集预处理主流程。

    参数:
      augment_copies: 每个训练样本生成的 wrist-style 增强副本数 (0=不增强)
    """
    print("=" * 60)
    print("Microsoft RecoFit 数据集预处理 — Domain Adaptation 版本")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    mat_path = os.path.join(DATA_DIR, MAT_FILENAME)

    if not os.path.exists(mat_path):
        print(f"\n错误: 找不到 {mat_path}")
        print("请先下载 exercise_data.50.0000_multionly.mat 到 data/raw/recofit/")
        sys.exit(1)

    # ================================================================
    # Step 1: 加载 + 提取目标类别片段
    # ================================================================
    print(f"\n[1/7] 加载 {mat_path} ...")
    t_start = time.time()
    mat_data = loadmat(mat_path)
    print(f"  加载耗时: {time.time() - t_start:.1f}s")

    print("  提取目标类别活动片段...")
    segments, subjects_all = extract_segments(mat_data)
    print(f"  提取活动片段: {len(segments)}")

    if len(segments) == 0:
        print("错误: 未提取到任何目标类别片段！")
        print("请检查 RECOFIT_TO_TARGET 映射是否正确。")
        sys.exit(1)

    # ================================================================
    # Step 2: 滑动窗口
    # ================================================================
    print(f"\n[2/7] 滑动窗口切分 (窗口={WINDOW_SIZE}, 步长={STEP_SIZE})...")
    X, y, subjects_win = sliding_window(segments, WINDOW_SIZE, STEP_SIZE)
    print(f"  窗口总数: {X.shape[0]}, 形状: {X.shape}")

    # 类别分布统计
    unique_labels, counts = np.unique(y, return_counts=True)
    print("  各类别窗口数:")
    for lbl, cnt in zip(unique_labels, counts):
        name = TARGET_CANONICAL_NAMES[lbl]
        print(f"    [{lbl}] {name}: {cnt}")

    # ================================================================
    # Step 3: 按受试者划分 train/test
    # ================================================================
    print(f"\n[3/7] 按受试者划分 train/test (test_ratio={TEST_RATIO})...")
    unique_subjs = np.unique(subjects_win)
    train_subjs, test_subjs = train_test_split(
        unique_subjs, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    print(f"  训练受试者: {len(train_subjs)}, 测试受试者: {len(test_subjs)}")

    train_mask = np.isin(subjects_win, train_subjs)
    test_mask = np.isin(subjects_win, test_subjs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  训练集: {X_train.shape[0]} 窗口")
    print(f"  测试集: {X_test.shape[0]} 窗口")

    # 检查测试集是否包含所有类别
    test_classes = set(np.unique(y_test))
    missing = set(range(len(TARGET_CANONICAL_NAMES))) - test_classes
    if missing:
        print(f"  ⚠ 测试集缺少类别: {[TARGET_CANONICAL_NAMES[c] for c in missing]}")
        # 从训练集强制分配至少1个样本到测试集
        for cls in missing:
            cls_train_idx = np.where(y_train == cls)[0]
            if len(cls_train_idx) > 1:
                move_idx = cls_train_idx[-1]
                X_test = np.concatenate([X_test, X_train[move_idx:move_idx+1]], axis=0)
                y_test = np.concatenate([y_test, y_train[move_idx:move_idx+1]])
                X_train = np.delete(X_train, move_idx, axis=0)
                y_train = np.delete(y_train, move_idx)
                print(f"    已强制分配 {TARGET_CANONICAL_NAMES[cls]} 到测试集")

    # ================================================================
    # Step 4: PCA 方向归一化
    # ================================================================
    print(f"\n[4/7] PCA 方向归一化 (YZ平面 → pc1 + mag)...")
    t_start = time.time()
    X_train = batch_pca_orientation_normalize(X_train)
    X_test = batch_pca_orientation_normalize(X_test)
    print(f"  PCA 耗时: {time.time() - t_start:.1f}s")
    print(f"  新通道: [acc_x, acc_pc1, acc_mag, gyr_x, gyr_pc1, gyr_mag]")

    # ================================================================
    # Step 5: Z-score 归一化
    # ================================================================
    print(f"\n[5/7] Z-score 归一化 (仅在训练集拟合 mean/std)...")
    mean, std = compute_zscore_params(X_train)
    ch_names = ['acc_x', 'acc_pc1', 'acc_mag', 'gyr_x', 'gyr_pc1', 'gyr_mag']
    print("  Per-channel 统计量:")
    for c in range(6):
        print(f"    {ch_names[c]:12s}  mean={mean[0, c, 0]:+.4f}  std={std[0, c, 0]:.4f}")

    X_train = apply_zscore(X_train, mean, std)
    X_test = apply_zscore(X_test, mean, std)

    # 归一化后统计
    print("  归一化后 train 统计:")
    for c in range(6):
        print(f"    {ch_names[c]:12s}  mean={X_train[:, c, :].mean():+.4f}  "
              f"std={X_train[:, c, :].std():.4f}  "
              f"range=[{X_train[:, c, :].min():+.2f}, {X_train[:, c, :].max():+.2f}]")

    # ================================================================
    # Step 6: 数据增强 (wrist-style)
    # ================================================================
    if augment_copies > 0:
        print(f"\n[6/7] Wrist-style 数据增强 (生成 {augment_copies}x 增强副本)...")
        X_aug, y_aug = generate_augmented_samples(X_train, y_train, num_copies=augment_copies)
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug])
        print(f"  增强后训练集: {X_train.shape[0]} 窗口 (含 {augment_copies}x 增强样本)")
    else:
        print(f"\n[6/7] 跳过数据增强 (--augment 0)")

    # ================================================================
    # Step 7: 整理并保存
    # ================================================================
    print(f"\n[7/7] 保存处理后的数据...")

    # Domain label: 所有 RecoFit 样本均为 forearm (domain_id=0)
    domain_train = np.zeros(len(X_train), dtype=np.int64)
    domain_test = np.zeros(len(X_test), dtype=np.int64)

    # 最终统计
    print(f"\n{'='*60}")
    print(f"最终数据统计:")
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"  类别数: {len(TARGET_CANONICAL_NAMES)}")
    print(f"  各类别训练样本数:")
    for lbl in range(len(TARGET_CANONICAL_NAMES)):
        cnt_tr = (y_train == lbl).sum()
        cnt_te = (y_test == lbl).sum()
        print(f"    [{lbl}] {TARGET_CANONICAL_NAMES[lbl]:25s}  train={cnt_tr:5d}  test={cnt_te:5d}")
    print(f"  Domain: forearm (id={DOMAIN_ID})")

    # 打乱训练集
    rng = np.random.RandomState(RANDOM_SEED)
    train_idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    domain_train = domain_train[train_idx]

    # 保存到 data/processed/ (与 merge_datasets.py 兼容)
    print(f"\n保存文件到 {OUT_DIR}...")
    np.save(os.path.join(OUT_DIR, 'recofit_x_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'recofit_y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'recofit_x_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'recofit_y_test.npy'), y_test)
    np.save(os.path.join(OUT_DIR, 'recofit_domain_train.npy'), domain_train)
    np.save(os.path.join(OUT_DIR, 'recofit_domain_test.npy'), domain_test)

    # 保存标签映射
    mapping_df = pd.DataFrame([
        (lbl, name) for lbl, name in enumerate(TARGET_CANONICAL_NAMES)
    ], columns=['label', 'name'])
    mapping_df.to_csv(os.path.join(OUT_DIR, 'recofit_label_mapping.csv'), index=False)

    # 保存归一化参数（推理时复用）
    np.savez(os.path.join(OUT_DIR, 'recofit_norm_params.npz'),
             mean=mean.squeeze(), std=std.squeeze(),
             norm_type='zscore', channels=ch_names)

    # 保存 domain 元信息
    domain_meta = {
        'domain_id': DOMAIN_ID,
        'sensor_position': SENSOR_POSITION,
        'domain_label': 0,
        'description': 'arm-band on forearm (PCA orientation normalized + Z-score)',
        'preprocessing': {
            'pca_orientation_norm': True,
            'normalization': 'zscore',
            'target_classes': TARGET_CANONICAL_NAMES,
            'augment_copies': augment_copies,
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
        },
    }
    with open(os.path.join(OUT_DIR, 'recofit_domain_info.json'), 'w', encoding='utf-8') as f:
        json.dump(domain_meta, f, indent=2, ensure_ascii=False)

    # 额外保存一份到子目录 (向后兼容)
    sub_out_dir = 'data/processed/recofit/'
    os.makedirs(sub_out_dir, exist_ok=True)
    np.save(os.path.join(sub_out_dir, 'x_train.npy'), X_train)
    np.save(os.path.join(sub_out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(sub_out_dir, 'domain_train.npy'), domain_train)
    np.save(os.path.join(sub_out_dir, 'x_test.npy'), X_test)
    np.save(os.path.join(sub_out_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(sub_out_dir, 'domain_test.npy'), domain_test)
    mapping_df.to_csv(os.path.join(sub_out_dir, 'label_mapping.csv'), index=False)
    with open(os.path.join(sub_out_dir, 'domain_info.json'), 'w') as f:
        json.dump(domain_meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("✅ RecoFit Domain Adaptation 预处理完成！")
    print(f"   输出: {OUT_DIR}recofit_*.npy")
    print(f"   类别: {len(TARGET_CANONICAL_NAMES)} 个 (与 wrist 数据集重叠)")
    print(f"   归一化: PCA 方向归一化 + Z-score")
    print(f"   增强: {augment_copies}x wrist-style 增强")
    print(f"{'='*60}")


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RecoFit 数据集预处理 — Domain Adaptation 版本'
    )
    parser.add_argument('--augment', type=int, default=0,
                        help='每个训练样本生成的 wrist-style 增强副本数 (默认: 0)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'原始数据目录 (默认: {DATA_DIR})')
    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help=f'输出目录 (默认: {OUT_DIR})')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir

    preprocess(augment_copies=args.augment)
