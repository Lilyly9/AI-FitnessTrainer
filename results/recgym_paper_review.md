# RecGym 数据集论文调研报告

---

## 1. RecGym 数据集基本信息

| 项目 | 内容 |
|------|------|
| 数据集名称 | RecGym: Gym Workouts Recognition Dataset with IMU and Capacitive Sensor |
| 来源 | UCI Machine Learning Repository |
| DOI | [10.24432/C5PW4K](https://doi.org/10.24432/C5PW4K) |
| 发布时间 | 2025年2月18日 |
| 受试者 | 10人 (5男5女) |
| 会话数 | 50次 (每人5次, 每次约1小时) |
| 总数据量 | ~4,432,070条 |
| 采样率 | 20 Hz |
| 传感器位置 | 手腕(Wrist)、口袋(Pocket)、小腿(Calf) |
| 传感器类型 | IMU (6通道: Acc_x/y/z + Gyro_x/y/z) + 人体电容HBC (1通道: C_1) |
| 动作类别 | 12类: Adductor, ArmCurl, BenchPress, LegCurl, LegPress, Riding, RopeSkipping, Running, Squat, StairsClimber, Walking, Null(休息过渡) |

### 关键特点

- **双模态传感**: IMU + 人体电容(HBC), HBC通过人体静电耦合感知跨身体部位的运动关联
- **多佩戴位置**: 手腕/口袋/小腿三个位置同时采集
- **真实健身场景**: 自由重量训练, 含哑铃、器械等多种运动
- **含Null类**: 包含组间休息等过渡状态, 更贴近实际应用

---

## 2. 重点论文分析

### 论文1 (核心): Multi-Task Learning for Sports Training Monitoring via Multimodal Fusion of IMU and Human-Body Capacitance Signals

| 项目 | 内容 |
|------|------|
| 作者 | Wang Hailong, Guan Xinru, Jiang Sheng, Zhang Lijun |
| 单位 | 佳木斯大学 (体育学院 + 信息与电子技术学院) |
| 期刊 | Journal of Applied Science and Engineering (JASE) |
| 时间 | 2025年12月投稿, 2026年1月接收, 2026年3月发表 |
| 开源 | CC BY 4.0 |
| 链接 | [DOAJ](https://doaj.org/article/1019881d9a6f4de79f8d288b92507fc2), [JASE](https://jase.tku.edu.tw/jase/?tkuisotope=multi-task-learning-for-sports-training-monitoring-via-multimodal-fusion-of-imu-and-human-body-capacitance-signals) |

#### 模型架构: GAT-TCN

1. **Graph Attention Network (GAT)**:
   - 将每个IMU通道(6个)和HBC通道(1个)建模为图节点
   - 结合物理先验与数据驱动定义边集(如加速度计与陀螺仪之间的互补关系)
   - 通过多头注意力机制自适应学习跨通道、跨模态的依赖关系

2. **Temporal Convolutional Network (TCN)**:
   - 在GAT提取的通道融合特征基础上
   - 使用膨胀卷积提取多尺度时序节奏特征
   - 捕获不同频率的运动模式(快速爆发动作 vs 慢速控制动作)

3. **多任务联合学习**:
   - 共享特征提取层 + 三个独立任务头
   - 联合优化: 动作识别 (CrossEntropy) + 次数计数 (MSE) + 强度估计 (MSE)

#### 实验结果

| 任务 | 指标 | 结果 |
|------|------|------|
| 动作识别 | Accuracy (LOSO) | **0.894** |
| 动作识别 | Kappa | 0.871 |
| 重复次数计数 | MAE | 0.234 |
| 训练强度估计 | MAE | 0.156 |

#### 核心结论

- **HBC对低幅度运动有显著提升**: 轻哑铃动作、小范围关节调整等场景下, 融合模型准确率稳定在~0.9, 而纯IMU模型出现明显下降
- **多任务学习比单任务更优**: 三个任务共享表征可以互相促进
- **GAT自适应通道融合优于简单的concat/加法融合**

#### 局限性

- 仅在一个数据集(RecGym)上验证, 泛化性待考察
- GAT-TCN计算量较大, 未讨论实时推理延迟
- 人体电容传感器的硬件普及度低, 实际部署受限
- 10人留一法虽严格, 但样本量有限

---

### 论文2: Hybrid CNN-Dilated Self-attention Model Using Inertial and Body-Area Electrostatic Sensing for Gym Workout Recognition, Counting, and User Authentification

| 项目 | 内容 |
|------|------|
| 作者 | Sizhen Bian, Vitor Fortes Rey, Siyu Yuan, Paul Lukowicz |
| 单位 | DFKI (德国人工智能研究中心), Kaiserslautern |
| 发表 | arXiv: 2503.06311, 2025年3月 |
| 链接 | [arXiv](https://arxiv.org/abs/2503.06311) |

#### 模型架构: CNN + Dilated Self-Attention

1. **CNN特征提取层**: 多尺度1D卷积, 提取局部时序模式
2. **Dilated Self-Attention**: 在CNN特征之上应用膨胀自注意力, 扩大感受野捕获长程依赖
3. **多任务输出头**: 动作识别 + 次数计数 + 用户认证

#### 预处理方式

- 滑动窗口: **4秒 (80个读数, 20Hz)**, 重叠2秒
- 对比模型: Random Forest, DeepConvLSTM, ResNet21

#### 实验结果

| 场景 | 指标 | 结果 |
|------|------|------|
| 手腕 IMU+HBC | F-score | **0.93-0.94** |
| 手腕 IMU-only | F-score | 略低于融合 |
| HBC-only | F-score | 0.25-0.45 |

#### 核心发现

- **HBC对动作分类的提升是边际的**(slight boost), 但对**重复次数计数的提升是实质性的**(substantial advantage)
- HBC单独使用时性能很差(0.25-0.45 F-score), 必须与IMU配合
- 人体电容传感器功耗极低(μW级), 适合可穿戴设备

#### 局限性

- 仅在可控健身房环境下测试
- HBC传感器目前是研究原型, 无商业产品
- 用户认证任务的数据量较少, 结论可靠性有限

---

### 论文3: Bridging Generalization and Personalization in Human Activity Recognition via On-Device Few-Shot Learning

| 项目 | 内容 |
|------|------|
| 作者 | Kang, Moosmann, Liu, Zhou, Magno, Lukowicz & Bian |
| 单位 | DFKI / ETH Zurich |
| 发表 | MobileHCI 2025, 2025年9月 |
| 链接 | [DFKI](https://www.dfki.de/en/web/research/projects-and-publications/publication/16133) |

#### 模型架构: 1D-CNN + On-Device Few-Shot

1. **通用特征提取器**: 1D-CNN with residual blocks, 在所有用户数据上预训练(冻结)
2. **个性化分类器**: 仅更新最后的全连接分类层
3. **部署平台**: RISC-V GAP9 微控制器 (超低功耗边缘AI芯片)

#### 实验结果 (RecGym)

| 指标 | 结果 |
|------|------|
| 个性化后准确率提升 | **+3.73%** |
| 推理延迟 | 0.34 ms/sample |
| 训练更新延迟 | 0.07-0.17 ms |
| 单次更新能耗 | ~4 μJ |
| 能效比 | STM32F7的250倍 |

#### 核心创新

- 将HAR模型部署到极低功耗边缘设备
- 通过few-shot learning实现设备端个性化, 无需上传隐私数据

#### 局限性

- 仅更新分类层, 特征提取器可能无法适应全新动作类型
- GAP9芯片目前不普及, 商业化尚早

---

## 3. 三篇论文对比总结

| 维度 | 论文1 (GAT-TCN) | 论文2 (CNN+SelfAttn) | 论文3 (Few-Shot) |
|------|:--:|:--:|:--:|
| 准确率 | 89.4% (LOSO) | F-score 0.93-0.94 | 提升3.73% (含基础) |
| 模态 | IMU + HBC | IMU + HBC | IMU (可扩展HBC) |
| 模型复杂度 | 高 (GAT+TCN) | 中 (CNN+Attention) | 低 (1D-CNN残差块) |
| 多任务 | 识别+计数+强度 | 识别+计数+认证 | 仅识别 |
| 部署友好 | 未讨论 | 未讨论 | 极优(GAP9 MCU) |
| 创新点 | 图注意力通道融合 | HBC对计数的贡献 | 设备端个性化 |
| 窗口大小 | 未明确 | 4秒(80读数) | 未明确 |

---

## 4. 对我们项目的参考价值

### 可借鉴的方法

1. **滑动窗口参考值**: 论文2使用4秒窗口(80读数@20Hz), 对应我们100Hz的400点窗口; 我们当前200点窗口(2秒)偏短, 可考虑增加到300-400
2. **多任务学习思路**: 论文1的联合优化(识别+计数+强度)可以借鉴, 我们已有分类器和rep_counter, 可以共享特征提取器
3. **通道注意力机制**: 论文1的GAT通道融合可以简化 — 在我们6通道IMU上使用Squeeze-and-Excitation或简单注意力模块

### 对我们模型的改进方向

1. 当前纯IMU 6通道 → 可以引入通道注意力提升对关键通道的关注
2. 当前单任务(分类) → 可以联合训练分类+计数任务
3. 当前模型43K参数 → 适合边缘部署, 可以尝试导出ONNX

---

## 5. 参考文献

1. Wang H, Guan X, Jiang S, Zhang L. "Multi-Task Learning for Sports Training Monitoring via Multimodal Fusion of IMU and Human-Body Capacitance Signals." *JASE*, 2026. DOI: [10.24432/C5PW4K](https://doi.org/10.24432/C5PW4K)

2. Bian S, Rey VF, Yuan S, Lukowicz P. "Hybrid CNN-Dilated Self-attention Model Using Inertial and Body-Area Electrostatic Sensing for Gym Workout Recognition, Counting, and User Authentification." *arXiv:2503.06311*, 2025.

3. Kang et al. "Bridging Generalization and Personalization in Human Activity Recognition via On-Device Few-Shot Learning." *MobileHCI 2025*, 2025.

4. Bian S, Rey VF, Yuan S, Lukowicz P. "The Contribution of Human Body Capacitance/Body-Area Electric Field To Individual and Collaborative Activity Recognition." *ABC 2025*, 2025.
