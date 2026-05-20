"""
MM-Fit 数据集预处理
================================================================
来源: https://zenodo.org/records/7672767
规模: 20 sessions × 10 类健身动作
传感器: 左手手表 IMU (加速度计 + 陀螺仪), 100Hz
动作 (10 类):
  squats(0), pushups(1), dumbbell_shoulder_press(2), lunges(3),
  dumbbell_rows(4), situps(5), tricep_extensions(6), bicep_curls(7),
  lateral_shoulder_raises(8), jumping_jacks(9)

原始数据格式:
  - {session}_sw_l_acc.npy: (T, 5), 列 [t_ms, t_ns, x, y, z]
  - {session}_sw_l_gyr.npy: (T, 5), 列 [t_ms, t_ns, x, y, z]
  - {session}_labels.csv: 无表头 [start_frame, end_frame, label_id, activity]

使用方法:
  python src/datasets/preprocess_mmfit.py --train-sessions 16

输出: data/processed/mmfit_x_train.npy, mmfit_y_train.npy,
               mmfit_x_test.npy, mmfit_y_test.npy
"""

import numpy as np
import pandas as pd
import os
import argparse
from collections import Counter
from glob import glob

# ==================== 配置 ====================
DATA_DIR = 'data/raw/mm-fit/'
OUT_DIR = 'data/processed/'
DATASET_NAME = 'mmfit'
WINDOW_SIZE = 200
STEP_SIZE = 100
SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

LABEL_MAP = {
    'squats': 0,
    'pushups': 1,
    'dumbbell_shoulder_press': 2,
    'lunges': 3,
    'dumbbell_rows': 4,
    'situps': 5,
    'tricep_extensions': 6,
    'bicep_curls': 7,
    'lateral_shoulder_raises': 8,
    'jumping_jacks': 9,
}

DOMAIN_ID = 'wrist'  # MM-Fit 传感器佩戴位置：左手手腕 (smartwatch)

DEFAULT_TRAIN_SESSIONS = [
    'w00', 'w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07',
    'w08', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15',
]
DEFAULT_TEST_SESSIONS = ['w16', 'w17', 'w18', 'w19']


def extract_session(session_dir, session_name):
    """从单 session 的原始 .npy 文件中提取 6 通道 IMU 片段。

    Args:
        session_dir: session 目录路径
        session_name: 如 'w00'

    Returns:
        list[dict]: 每个片段 {'data': (T,6), 'label': int}
    """
    acc_file = os.path.join(session_dir, f'{session_name}_sw_l_acc.npy')
    gyr_file = os.path.join(session_dir, f'{session_name}_sw_l_gyr.npy')
    label_file = os.path.join(session_dir, f'{session_name}_labels.csv')

    if not os.path.exists(acc_file) or not os.path.exists(gyr_file):
        print(f"  跳过 {session_name}: 缺少传感器文件")
        return []
    if not os.path.exists(label_file):
        print(f"  跳过 {session_name}: 缺少标签文件")
        return []

    acc_data = np.load(acc_file)  # (T, 5)
    gyr_data = np.load(gyr_file)  # (T, 5)

    df_label = pd.read_csv(label_file, header=None,
                           names=['start_frame', 'end_frame', 'label_id', 'activity'])

    segments = []
    for _, row in df_label.iterrows():
        start_f = int(row['start_frame'])
        end_f = int(row['end_frame'])
        activity = str(row['activity']).strip().lower()

        if activity not in LABEL_MAP:
            continue

        if start_f < 0 or end_f >= acc_data.shape[0] or end_f <= start_f:
            continue

        # 跳过前两列时间戳，取 x/y/z
        acc_seg = acc_data[start_f:end_f + 1, 2:5]
        gyr_seg = gyr_data[start_f:end_f + 1, 2:5]

        if acc_seg.shape[0] < WINDOW_SIZE:
            continue

        combined = np.column_stack([acc_seg, gyr_seg]).astype(np.float32)
        segments.append({
            'data': combined,
            'label': LABEL_MAP[activity],
            'session': session_name,
        })

    return segments


def sliding_window(segments, window_size, step_size):
    """从片段列表中创建滑动窗口。

    Args:
        segments: list of {'data': (T,6), 'label': int}
        window_size: 窗口大小
        step_size: 步长

    Returns:
        X: (N, 6, window_size), y: (N,), sessions: (N,) session名称数组
    """
    X_list, y_list, session_list = [], [], []

    for seg in segments:
        data = seg['data']
        label = seg['label']
        length = data.shape[0]

        for start in range(0, length - window_size + 1, step_size):
            end = start + window_size
            window = data[start:end].T  # (6, window_size)
            if np.isnan(window).any():
                continue
            X_list.append(window)
            y_list.append(label)
            session_list.append(seg.get('session', 'unknown'))

    if not X_list:
        return (np.empty((0, 6, window_size), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.array([], dtype=str))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    sessions = np.array(session_list)
    return X, y, sessions


def process_from_csv(csv_path, train_sessions, test_sessions):
    """从 mmfit_processed.csv 处理（已有提取后的 CSV）。

    注意: CSV 中的 subject_id 列名代表 session (如 'w00')。
    """
    print(f"从 CSV 读取: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  总行数: {len(df)}, 列: {df.columns.tolist()}")

    if 'subject_id' not in df.columns:
        raise ValueError("CSV 缺少 subject_id 列")

    sessions_in_csv = sorted(df['subject_id'].unique())
    print(f"  可用 session: {sessions_in_csv}")

    # 收集片段
    segments = []
    for session in sessions_in_csv:
        if session not in train_sessions and session not in test_sessions:
            print(f"  警告: {session} 不在训练/测试集中，跳过")
            continue

        session_df = df[df['subject_id'] == session]
        labels = session_df['label'].values
        data = session_df[SENSOR_COLS].values.astype(np.float32)

        for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = data[start:end].T
            if np.isnan(window).any():
                continue
            label = int(Counter(labels[start:end]).most_common(1)[0][0])
            segments.append({
                'data_window': window,
                'label': label,
                'session': session,
            })

    # 按 session 拆分
    X_train, y_train = [], []
    X_test, y_test = [], []

    for seg in segments:
        w, l, s = seg['data_window'], seg['label'], seg['session']
        if s in train_sessions:
            X_train.append(w)
            y_train.append(l)
        elif s in test_sessions:
            X_test.append(w)
            y_test.append(l)

    X_train = np.stack(X_train).astype(np.float32) if X_train else np.empty(
        (0, 6, WINDOW_SIZE), dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.stack(X_test).astype(np.float32) if X_test else np.empty(
        (0, 6, WINDOW_SIZE), dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    return X_train, y_train, X_test, y_test


def process_from_raw(raw_dir, train_sessions, test_sessions):
    """从原始 .npy 文件处理。"""
    print(f"从原始数据目录处理: {raw_dir}")

    all_session_dirs = sorted(glob(os.path.join(raw_dir, 'w*')))
    if not all_session_dirs:
        print(f"错误: {raw_dir} 中没有找到 session 目录 (w00-w19)")
        return None

    available_sessions = [os.path.basename(d) for d in all_session_dirs]
    print(f"发现 {len(available_sessions)} 个 session: {available_sessions}")

    train_parts, test_parts = [], []

    for session_dir in all_session_dirs:
        session_name = os.path.basename(session_dir)

        if session_name not in train_sessions and session_name not in test_sessions:
            print(f"  跳过 {session_name}: 不在训练/测试集划分中")
            continue

        is_train = session_name in train_sessions
        print(f"  处理 {session_name} ({'训练' if is_train else '测试'})...")

        segments = extract_session(session_dir, session_name)
        if not segments:
            print(f"    警告: 未提取到片段")
            continue

        X, y, sessions = sliding_window(segments, WINDOW_SIZE, STEP_SIZE)
        print(f"    提取 {len(X)} 个窗口, 类别分布: {dict(sorted(Counter(y).items()))}")

        if is_train:
            train_parts.append((X, y))
        else:
            test_parts.append((X, y))

    if not train_parts:
        print("错误: 没有训练数据")
        return None

    X_train = np.concatenate([p[0] for p in train_parts], axis=0)
    y_train = np.concatenate([p[1] for p in train_parts], axis=0)

    if test_parts:
        X_test = np.concatenate([p[0] for p in test_parts], axis=0)
        y_test = np.concatenate([p[1] for p in test_parts], axis=0)
    else:
        X_test = np.empty((0, 6, WINDOW_SIZE), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.int64)

    return X_train, y_train, X_test, y_test


def preprocess(csv_fallback=True, train_sessions=None, test_sessions=None):
    """主预处理流程。"""
    if train_sessions is None:
        train_sessions = DEFAULT_TRAIN_SESSIONS
    if test_sessions is None:
        test_sessions = DEFAULT_TEST_SESSIONS

    print("=" * 60)
    print("MM-Fit 数据集预处理")
    print("=" * 60)
    print(f"训练 session: {train_sessions}")
    print(f"测试 session: {test_sessions}")
    print(f"窗口大小: {WINDOW_SIZE}, 步长: {STEP_SIZE}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)

    raw_dir = DATA_DIR
    if os.path.isdir(raw_dir) and glob(os.path.join(raw_dir, 'w*')):
        # 原始 .npy 文件路径
        result = process_from_raw(raw_dir, train_sessions, test_sessions)
    elif csv_fallback:
        csv_path = os.path.join(OUT_DIR, 'mmfit_processed.csv')
        if os.path.exists(csv_path):
            print("原始 .npy 文件未找到，使用 mmfit_processed.csv 作为输入...")
            print("提示: 如需从原始数据重新处理，请从 Zenodo 下载:")
            print("  https://zenodo.org/records/7672767")
            print("  解压到 data/raw/mm-fit/ (包含 w00-w19 子目录)")
            print()
            result = process_from_csv(csv_path, train_sessions, test_sessions)
        else:
            print(f"错误: 未找到原始数据 ({raw_dir})，也未找到 {csv_path}")
            print()
            print("请先下载 MM-Fit 数据集:")
            print("  1. 访问 https://zenodo.org/records/7672767")
            print("  2. 下载所有 session 的 ZIP 文件 (w00-w19)")
            print("  3. 解压到 data/raw/mm-fit/")
            print()
            print("预期目录结构:")
            print("  data/raw/mm-fit/")
            print("    w00/")
            print("      w00_sw_l_acc.npy")
            print("      w00_sw_l_gyr.npy")
            print("      w00_labels.csv")
            print("    w01/")
            print("      ...")
            return
    else:
        print(f"错误: 数据目录 {raw_dir} 不存在")
        return

    if result is None:
        return

    X_train, y_train, X_test, y_test = result

    # ---- Min-Max 归一化（基于训练集） ----
    print(f"\nMin-Max 归一化...")
    min_vals = X_train.min(axis=(0, 2), keepdims=True)
    max_vals = X_train.max(axis=(0, 2), keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-8] = 1.0
    X_train = (X_train - min_vals) / range_vals
    if len(X_test) > 0:
        X_test = (X_test - min_vals) / range_vals

    # ---- 统计 ----
    print(f"\n训练集: X={X_train.shape}, y={y_train.shape}")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    print("  类别分布 (训练):")
    for lbl, cnt in zip(train_unique, train_counts):
        name = [k for k, v in LABEL_MAP.items() if v == lbl][0]
        print(f"    {lbl} ({name}): {cnt} 窗口")

    if len(X_test) > 0:
        print(f"测试集: X={X_test.shape}, y={y_test.shape}")
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        print("  类别分布 (测试):")
        for lbl, cnt in zip(test_unique, test_counts):
            name = [k for k, v in LABEL_MAP.items() if v == lbl][0]
            print(f"    {lbl} ({name}): {cnt} 窗口")

    # ---- 验证 ----
    print(f"\n验证...")
    assert X_train.ndim == 3 and X_train.shape[1:] == (6, WINDOW_SIZE), \
        f"训练集形状错误: {X_train.shape}, 期望 (N, 6, {WINDOW_SIZE})"
    assert X_train.dtype == np.float32, f"训练集 dtype 错误: {X_train.dtype}"
    assert len(np.unique(y_train)) <= len(LABEL_MAP), \
        f"训练集标签数量异常: {np.unique(y_train)}"

    if len(X_test) > 0:
        assert X_test.shape[1:] == (6, WINDOW_SIZE), \
            f"测试集形状错误: {X_test.shape}"
        assert X_test.dtype == np.float32

    # 归一化范围检查
    assert X_train.min() >= -0.01 and X_train.max() <= 1.01, \
        f"归一化范围异常: [{X_train.min():.4f}, {X_train.max():.4f}]"

    print("[OK] 所有验证通过")

    # ---- 保存 ----
    print(f"\n保存到 {OUT_DIR}...")
    np.save(os.path.join(OUT_DIR, f'{DATASET_NAME}_x_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, f'{DATASET_NAME}_y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, f'{DATASET_NAME}_x_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, f'{DATASET_NAME}_y_test.npy'), y_test)

    # 保存标签映射
    mapping_path = os.path.join(OUT_DIR, f'{DATASET_NAME}_label_mapping.csv')
    pd.DataFrame(
        [(lbl, name) for name, lbl in sorted(LABEL_MAP.items(), key=lambda x: x[1])],
        columns=['label_id', 'activity_name']
    ).to_csv(mapping_path, index=False)

    # Domain label: wrist (domain_id=1)
    domain_train = np.ones(len(X_train), dtype=np.int64)
    domain_test = np.ones(len(X_test), dtype=np.int64)
    np.save(os.path.join(OUT_DIR, f'{DATASET_NAME}_domain_train.npy'), domain_train)
    np.save(os.path.join(OUT_DIR, f'{DATASET_NAME}_domain_test.npy'), domain_test)

    # 保存归一化参数
    np.savez(os.path.join(OUT_DIR, f'{DATASET_NAME}_norm_params.npz'),
             min_vals=min_vals.squeeze(), max_vals=max_vals.squeeze())

    print(f"Domain label: {DOMAIN_ID} (id=1)")
    print(f"\n输出文件:")
    print(f"  {OUT_DIR}{DATASET_NAME}_x_train.npy  ({X_train.shape})")
    print(f"  {OUT_DIR}{DATASET_NAME}_y_train.npy  ({y_train.shape})")
    print(f"  {OUT_DIR}{DATASET_NAME}_x_test.npy   ({X_test.shape})")
    print(f"  {OUT_DIR}{DATASET_NAME}_y_test.npy   ({y_test.shape})")
    print(f"  {OUT_DIR}{DATASET_NAME}_label_mapping.csv")
    print(f"  {OUT_DIR}{DATASET_NAME}_norm_params.npz")
    print("\n[OK] MM-Fit 预处理完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MM-Fit 数据集预处理')
    parser.add_argument('--train-sessions', type=int, default=16,
                        help='训练 session 数量 (前 N 个 session 作为训练集, 默认 16)')
    parser.add_argument('--no-csv-fallback', action='store_true',
                        help='禁用 CSV fallback, 仅从原始数据读取')
    parser.add_argument('--data-dir', default=None,
                        help='原始数据目录 (默认: data/raw/mm-fit/)')
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = args.data_dir

    # 动态划分 session
    all_sessions = [f'w{i:02d}' for i in range(20)]
    n_train = args.train_sessions
    train_sessions = all_sessions[:n_train]
    test_sessions = all_sessions[n_train:]
    print(f"Session 划分: 训练={n_train}, 测试={20 - n_train}")

    preprocess(
        csv_fallback=not args.no_csv_fallback,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
    )
