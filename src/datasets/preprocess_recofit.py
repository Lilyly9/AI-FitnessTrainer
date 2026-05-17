"""
Microsoft RecoFit 数据集预处理（.mat 版本）
================================================================
来源: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
规模: 200+ 人，多种健身房动作
传感器: 加速度计 + 陀螺仪 (arm-band)
格式: .mat (MATLAB) → 统一 .npy

使用方法:
  1. 下载 exercise_data.50.0000_multionly.mat 放到 data/raw/recofit/
  2. python src/datasets/preprocess_recofit.py

输出: data/processed/recofit/x_train.npy, y_train.npy, x_test.npy, y_test.npy
      以及 label_mapping.csv, norm_params.npz
"""

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
DATA_DIR = 'data/raw/recofit/'
OUT_DIR = 'data/processed/recofit/'
MAT_FILENAME = 'exercise_data.50.0000_multionly.mat'
WINDOW_SIZE = 200
STEP_SIZE = 20         # 步长 20，确保每类样本数 ≥ 1500
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ==================== 数据量最大的 25 个动作 ====================
SELECTED_ACTIVITIES = [
    'Elliptical machine',
    'Running (treadmill)',
    'Rowing machine',
    'Fast Alternating Punches',
    'Plank',
    'Seated Back Fly',
    'Lunge (alternating both legs, weight optional)',
    'Squat (arms in front of body, parallel to ground)',
    'Burpee',
    'Triceps Kickback (knee on bench) (label spans both arms)',
    'Two-arm Dumbbell Curl (both arms, not alternating)',
    'Dumbbell Row (knee on bench) (label spans both arms)',
    'Sit-up (hands positioned behind head)',
    'Sit-ups',
    'V-up',
    'Wall Squat',
    'Walking lunge',
    'Dumbbell Deadlift Row',
    'Bicep Curl',
    'Overhead Triceps Extension',
    'Russian Twist',
    'Crunch',
    'Shoulder Press (dumbbell)',
    'Dip',
    'Squat',
]

SKIP_ACTIVITIES = {
    '<Initial Activity>', 'Tap Left Device', 'Tap Right Device',
    'Non-Exercise', 'Device on Table', 'Arm Band Adjustment',
}

LABEL_MAP = {name: i for i, name in enumerate(SELECTED_ACTIVITIES)}


def extract_segments(mat_data):
    """从 .mat 数据结构中提取活动片段（仅限选定动作）。"""
    subject_data = mat_data['subject_data']
    num_subjects = subject_data.shape[0]
    all_segments = []
    all_subjects = []

    for subj_idx in range(num_subjects):
        records = subject_data[subj_idx, 0]          # shape (N,)

        for rec_idx in range(records.shape[0]):
            record = records[rec_idx]
            act_mat_1d = record['activityStartMatrix']   # (K,)
            data_1d = record['data']                     # (K,)

            k = min(len(act_mat_1d), len(data_1d))
            if k == 0:
                continue

            # 受试者 ID
            subj_id = record['subjectID']
            while isinstance(subj_id, np.ndarray) and subj_id.size > 0:
                subj_id = subj_id.ravel()[0]
            try:
                subj_id = int(subj_id)
            except:
                pass

            for seg_idx in range(k):
                act_block = act_mat_1d[seg_idx]        # (M, 7)
                data_struct = data_1d[seg_idx]

                acc_full = data_struct['accelDataMatrix'][0, 0]   # (T, 4)
                gyr_full = data_struct['gyroDataMatrix'][0, 0]    # (T, 4)

                if acc_full.ndim != 2 or acc_full.shape[1] < 4:
                    continue
                if gyr_full.ndim != 2 or gyr_full.shape[1] < 4:
                    continue

                # 遍历该片段内的每个小活动
                for row_idx in range(act_block.shape[0]):
                    name_field = act_block[row_idx, 0]
                    while isinstance(name_field, np.ndarray) and name_field.size > 0:
                        name_field = name_field.ravel()[0]
                    act_name = str(name_field)

                    if act_name in SKIP_ACTIVITIES or act_name not in LABEL_MAP:
                        continue

                    # 根据帧号切割
                    seq_info = act_block[row_idx, -1]
                    start_seq = seq_info['startSequenceNumberMaster']
                    end_seq = seq_info['endSequenceNumberMaster']
                    while isinstance(start_seq, np.ndarray) and start_seq.size > 0:
                        start_seq = start_seq.ravel()[0]
                    while isinstance(end_seq, np.ndarray) and end_seq.size > 0:
                        end_seq = end_seq.ravel()[0]

                    start_idx = int(start_seq) - 1
                    end_idx = int(end_seq) - 1

                    # 裁剪到有效范围
                    if start_idx < 0:
                        start_idx = 0
                    if end_idx >= acc_full.shape[0]:
                        end_idx = acc_full.shape[0] - 1
                    if end_idx <= start_idx:
                        continue

                    acc_seg = acc_full[start_idx:end_idx + 1, 1:]   # (seg_len, 3)
                    gyr_seg = gyr_full[start_idx:end_idx + 1, 1:]

                    all_segments.append({
                        'acc': acc_seg,
                        'gyr': gyr_seg,
                        'label': LABEL_MAP[act_name],
                        'subject_id': subj_id,
                    })
                    all_subjects.append(subj_id)

    return all_segments, np.array(all_subjects)


def sliding_window(segments, window_size, step_size):
    X_list, y_list, subj_list = [], [], []

    for seg in segments:
        acc = seg['acc']
        gyr = seg['gyr']
        length = acc.shape[0]

        for start in range(0, length - window_size + 1, step_size):
            end = start + window_size
            win = np.concatenate([
                acc[start:end, :].T,
                gyr[start:end, :].T,
            ], axis=0)                          # (6, 200)
            X_list.append(win)
            y_list.append(seg['label'])
            subj_list.append(seg['subject_id'])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    subjects = np.array(subj_list)
    return X, y, subjects


def preprocess():
    print("=" * 50)
    print("Microsoft RecoFit 数据集预处理")
    print("=" * 50)

    os.makedirs(OUT_DIR, exist_ok=True)
    mat_path = os.path.join(DATA_DIR, MAT_FILENAME)

    if not os.path.exists(mat_path):
        print(f"错误: 找不到 {mat_path}")
        return

    # 1. 加载
    print(f"加载 {mat_path} ...")
    mat_data = loadmat(mat_path)
    segments, subjects = extract_segments(mat_data)
    print(f"提取活动片段: {len(segments)}")

    # 2. 滑动窗口
    print(f"滑动窗口切分 (窗口{ WINDOW_SIZE }, 步长{ STEP_SIZE })...")
    X, y, subjects_win = sliding_window(segments, WINDOW_SIZE, STEP_SIZE)
    print(f"窗口总数: {X.shape[0]}, 形状: {X.shape}")

    # 3. 按受试者划分
    unique_subjs = np.unique(subjects_win)
    train_subjs, test_subjs = train_test_split(
        unique_subjs, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    print(f"训练受试者: {len(train_subjs)}, 测试受试者: {len(test_subjs)}")

    train_mask = np.isin(subjects_win, train_subjs)
    test_mask = np.isin(subjects_win, test_subjs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 4. Min-Max 归一化
    print("Min-Max 归一化...")
    min_vals = X_train.min(axis=(0, 2), keepdims=True)
    max_vals = X_train.max(axis=(0, 2), keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8
    X_train = (X_train - min_vals) / range_vals
    X_test = (X_test - min_vals) / range_vals

    # 5. 统计
    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    print(f"每类最少样本数: {min_count}")
    if min_count < 1500:
        print("⚠️ 警告: 某些类别不足 1500 样本")
    else:
        print("✅ 每类样本数均 ≥ 1500")

    # 6. 保存
    print("\n保存 .npy 文件...")
    np.save(os.path.join(OUT_DIR, 'x_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'x_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

    mapping_df = pd.DataFrame(
        [(i, name) for i, name in enumerate(SELECTED_ACTIVITIES)],
        columns=['label_id', 'activity_name']
    )
    mapping_df.to_csv(os.path.join(OUT_DIR, 'label_mapping.csv'), index=False)

    np.savez(os.path.join(OUT_DIR, 'norm_params.npz'),
             min_vals=min_vals.squeeze(), max_vals=max_vals.squeeze())

    assert len(set(train_subjs) & set(test_subjs)) == 0, "受试者有交集！"
    print("\n✅ 预处理完成！")


if __name__ == '__main__':
    preprocess()
