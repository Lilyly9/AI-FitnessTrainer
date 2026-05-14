# AI-FitnessTrainer

基于1D卷积神经网络（1D CNN）的健身动作分类系统。

## 项目简介

从可穿戴设备IMU传感器（加速度计+陀螺仪）时间序列数据中，自动识别5种健身动作类型。

## 环境配置

- Python 3.9+
- conda 环境：`fitness`
- 主要依赖：torch, numpy, pandas, matplotlib, scikit-learn

```bash
pip install -r requirements.txt
```

## 数据说明

- 数据集：Gym Gesture Classification Dataset（IEEE DataPort）
- 传感器通道（6通道）：acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
- 动作类别（5类）：chest_fly(0), chest_press(1), lat_pulldown(2), seated_row(3), tricep_extension(4)
- 受试者：1~4 训练，5 测试（按受试者划分，避免数据泄露）

## 预处理流程

1. 列名标准化：将原始列名映射为规范名称
2. 标签映射：字符串标签 → 0~4 数字标签
3. 缺失值处理：前向填充（ffill）传感器通道
4. Min-Max 归一化（基于训练集计算参数，应用于测试集）
5. 滑动窗口：窗口 200，步长 100，按受试者单独切分
6. 输出：`data/processed/x_train.npy, y_train.npy, x_test.npy, y_test.npy`

## 模型结构

- 3层 1D CNN + InstanceNorm + 最大池化 + 全局平均/最大池化 + 全连接
- 输入形状：(batch, 6, 200)
- 输出：5 类健身动作

## 训练策略

- 损失函数：加权交叉熵 + 标签平滑
- 优化器：SGD + Nesterov动量 + 余弦暖重启
- 数据增强：Mixup + 噪声 + 缩放 + 时间偏移 + 时间扭曲
- 集成学习：3模型投票
- 批次大小：32，训练轮次：80

## 使用方法

```bash
# 1. 数据预处理
python src/data_preprocessing.py

# 2. 模型训练
python src/train.py

# 3. 模型演示
python src/demo.py
```

## 输出产物

- `data/processed/` — 预处理后的训练/测试集
- `models/best_model.pth` — 最佳单模型
- `models/model_seed*.pth` — 集成模型
- `results/training_curve.png` — 训练曲线
- `results/confusion_matrix.png` — 混淆矩阵
- `results/classification_report.txt` — 分类报告
