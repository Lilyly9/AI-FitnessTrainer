# AI-FitnessTrainer
Research on the Recognition and Evaluation of Body-building Movements.
# AI-FitnessTrainer - 成员B（数据预处理 & 模型训练）

## 环境配置
- Python 3.9
- conda 环境：`fitness`
- 主要依赖：pandas, numpy, torch

## 数据说明
- 数据集：PAMAP2 手腕 IMU
- 文件路径：`data/processed/pamap2_processed.csv`
- 传感器通道：acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z (共6通道)
- 动作类别（4类）：走路(4→0)、跑步(5→1)、骑行(6→2)、跳绳(24→3)
- 受试者划分：101~108 训练，109 测试（按受试者划分，避免数据泄露）

## 预处理流程
1. 缺失值处理：前向填充（ffill）传感器通道
2. 标签映射：{4:0, 5:1, 6:2, 24:3}
3. Min-Max 归一化（基于训练集计算，应用于测试集）
4. 滑动窗口：窗口大小 200，步长 100，按每个受试者单独切分
5. 超出窗口标签：取窗口内出现最多的标签
6. 输出文件：`data/processed/x_train.npy, y_train.npy, x_test.npy, y_test.npy`

## 模型结构
- 2层 1D CNN + 最大池化 + 全局平均池化 + 全连接
- 输入形状：(batch, 6, 200)
- 输出：4 类动作

## 训练配置
- 损失函数：交叉熵损失（CrossEntropyLoss）
- 优化器：Adam，学习率 0.001
- 批次大小：32
- 训练轮次：30 epoch
- 验证集：从训练集中随机抽取 20%
- 模型保存：验证准确率最高的模型保存为 `models/best_model.pth`

## 结果
- 最佳验证准确率：96.2%
- 测试集准确率（受试者109）：90.3%