"""
多数据集预处理模块。

每个数据集提供 preprocess() 函数，输出统一的 .npy 格式：
  x: (N, 6, 200) float32  — 6通道传感器窗口
  y: (N,)     int64    — 标签 0~n_classes-1

通道顺序统一为: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
"""
