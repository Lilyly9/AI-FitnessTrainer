"""
多数据集预处理模块。

每个数据集提供 preprocess() 函数，输出统一的 .npy 格式：
  x: (N, 6, 200) float32  — 6通道传感器窗口
  y: (N,)     int64    — 标签 0~n_classes-1

Gym Gesture / MM-Fit 通道顺序:
  [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

RecoFit 通道顺序 (PCA 方向归一化后):
  [acc_x, acc_pc1, acc_mag, gyr_x, gyr_pc1, gyr_mag]
  — YZ 平面做 PCA 分解，消除 forearm/wrist 佩戴角度差异
"""
