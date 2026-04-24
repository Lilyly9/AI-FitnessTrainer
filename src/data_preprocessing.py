import pandas as pd
import numpy as np
from collections import Counter
import os

DATA_PATH = 'data/processed/pamap2_processed.csv'  
WINDOW_SIZE = 200
STEP_SIZE = 100
SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABEL_COL = 'label'
ID_COL = 'subject_id'
OUT_DIR = 'data/processed/'

# 原始标签 → 0~4 的映射
LABEL_MAP = {4: 0, 5: 1, 6: 2, 24: 3}

#  读取数据
df = pd.read_csv(DATA_PATH)
df[SENSOR_COLS] = df[SENSOR_COLS].fillna(method='ffill')   # 用前一个有效值填充
# 标签映射
df['label_mapped'] = df[LABEL_COL].map(LABEL_MAP)
# 检查是否有未映射的标签
if df['label_mapped'].isna().any():
    print("警告：存在未映射的标签，请检查。")
    print(df[df['label_mapped'].isna()][LABEL_COL].unique())
    exit()

# 获取所有受试者 ID，自动划分训练/测试集（最后一个 ID 作为测试）
all_subjects = sorted(df[ID_COL].unique())
TRAIN_IDS = all_subjects[:-1]   # 除最后一个外的所有
TEST_IDS = [all_subjects[-1]]   # 最后一个
print(f"训练集受试者: {TRAIN_IDS}")
print(f"测试集受试者: {TEST_IDS}")

# 按受试者划分原始数据
train_df = df[df[ID_COL].isin(TRAIN_IDS)].copy()
test_df = df[df[ID_COL].isin(TEST_IDS)].copy()
print(f"训练集原始样本数: {len(train_df)}")
print(f"测试集原始样本数: {len(test_df)}")

# Min-Max 归一化
feature_cols = SENSOR_COLS
# 计算训练集每列的 min/max
min_vals = train_df[feature_cols].min()
max_vals = train_df[feature_cols].max()

# 检测常数通道（min == max）
const_channels = []
for col in feature_cols:
    if max_vals[col] - min_vals[col] < 1e-8:
        const_channels.append(col)
if const_channels:
    print(f"发现常数通道，将强制置零: {const_channels}")

def minmax_normalize(data, min_vals, max_vals):
    denom = max_vals - min_vals
    denom[denom < 1e-8] = 1.0          # 防止除零
    normalized = (data - min_vals) / denom
    # 对于 min==max 的列，直接设为 0（归一化后自然为 0）
    for col in const_channels:
        normalized[col] = 0.0
    return normalized

train_df[feature_cols] = minmax_normalize(train_df[feature_cols], min_vals, max_vals)
test_df[feature_cols] = minmax_normalize(test_df[feature_cols], min_vals, max_vals)

# 滑动窗口函数
def create_windows(group, window_size, step_size, sensor_cols, label_col='label_mapped'):
    data = group[sensor_cols].values
    labels = group[label_col].values
    windows = []
    window_labels = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end].T
        if np.isnan(window).any():
            continue        
        windows.append(window)
        # 窗口内出现最多的标签
        label_counts = Counter(labels[start:end])
        most_common_label = label_counts.most_common(1)[0][0]
        window_labels.append(most_common_label)
    return np.array(windows), np.array(window_labels)

def process_dataset(df, window_size, step_size, sensor_cols, id_col):
    all_windows = []
    all_labels = []
    for athlete_id, group in df.groupby(id_col):
        group = group.reset_index(drop=True)
        wins, labs = create_windows(group, window_size, step_size, sensor_cols)
        all_windows.append(wins)
        all_labels.append(labs)
        print(f"  受试者 {athlete_id}: 生成 {len(wins)} 个窗口")
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

print("处理训练集...")
x_train, y_train = process_dataset(train_df, WINDOW_SIZE, STEP_SIZE, SENSOR_COLS, ID_COL)
print("处理测试集...")
x_test, y_test = process_dataset(test_df, WINDOW_SIZE, STEP_SIZE, SENSOR_COLS, ID_COL)

print(f"x_train 形状: {x_train.shape}, y_train 形状: {y_train.shape}")
print(f"x_test 形状: {x_test.shape}, y_test 形状: {y_test.shape}")

#保存为 .npy
os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, 'x_train.npy'), x_train)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'x_test.npy'), x_test)
np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

print("数据预处理完成，文件已保存到 data/processed/")