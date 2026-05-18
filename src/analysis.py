"""
运动质量评分与异常检测模块。

补充 rep_counter.py 和 exercise_classifier.py 的功能:
  - 运动质量评分（一致性、平滑度、活动度）
  - 组数 (set) 自动分割
  - 异常动作预警

用法:
  from analysis import score_exercise_quality, segment_into_sets, detect_anomaly

依赖:
  - rep_counter: 动作计数
  - exercise_classifier: 动作分类
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
from collections import Counter


def compute_signal_energy(data):
    """计算多通道信号的合能量（优先使用加速度计通道 0-2）。"""
    if data.ndim == 1:
        return np.abs(data)
    n_ch = min(3, data.shape[1])
    return np.sqrt(np.sum(data[:, :n_ch] ** 2, axis=1))


def detect_peaks_and_valleys(signal_1d, distance=20, prominence=None):
    """检测信号的峰值和谷值。"""
    if prominence is None:
        prominence = np.std(signal_1d) * 0.3

    peaks, peak_props = signal.find_peaks(signal_1d, distance=distance, prominence=prominence)
    valleys, valley_props = signal.find_peaks(-signal_1d, distance=distance, prominence=prominence)
    return peaks, peak_props, valleys, valley_props


def segment_into_sets(rep_positions, max_rest_samples=2000):
    """根据间隔将 repeat 划分成组 (sets)。

    Args:
        rep_positions: rep 峰值位置列表
        max_rest_samples: 超过此间隔视为组间休息（默认 20秒 x 100Hz）

    Returns:
        list[dict]: 每组 {set_num, reps, positions}
    """
    if len(rep_positions) < 2:
        return [{'set_num': 1, 'reps': len(rep_positions), 'positions': list(rep_positions)}]

    intervals = np.diff(rep_positions)
    set_boundaries = [0] + np.where(intervals > max_rest_samples)[0].tolist() + [len(rep_positions) - 1]

    sets = []
    for i in range(len(set_boundaries) - 1):
        start = set_boundaries[i]
        end = set_boundaries[i + 1] + 1
        sets.append({
            'set_num': i + 1,
            'reps': end - start,
            'positions': rep_positions[start:end],
        })
    return sets


def extract_cycle_from_peaks(data, peaks, cycle_idx):
    """提取第 cycle_idx 个动作周期的数据片段。"""
    if cycle_idx >= len(peaks) - 1:
        return None
    start = max(0, peaks[cycle_idx])
    end = min(peaks[cycle_idx + 1], len(data) - 1)
    if end - start < 10:
        return None
    return data[start:end].copy()


def resample_cycle(cycle, target_len=200):
    """将动作周期重采样到固定长度。"""
    if cycle is None or len(cycle) < 2:
        return None
    t_old = np.linspace(0, 1, len(cycle))
    t_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, cycle.shape[1]), dtype=np.float32)
    for c in range(cycle.shape[1]):
        resampled[:, c] = np.interp(t_new, t_old, cycle[:, c])
    return resampled


def score_exercise_quality(data, rep_info=None):
    """对运动数据进行质量评分。

    Args:
        data: (N, C) 传感器数据
        rep_info: (可选) rep 检测结果，若为空则自动检测

    Returns:
        dict: {
            overall_score, consistency_score, smoothness_score,
            range_of_motion_score, rep_quality_scores, feedback
        }
    """
    # 检测 reps
    if rep_info is None:
        energy = compute_signal_energy(data)
        b, a = signal.butter(3, 0.1)
        energy_filtered = signal.filtfilt(b, a, energy)
        prominence = np.std(energy_filtered) * 0.5
        peaks, _ = signal.find_peaks(energy_filtered, distance=100, prominence=prominence)
        rep_positions = peaks.tolist()
    else:
        rep_positions = rep_info if isinstance(rep_info, list) else list(rep_info)

    if len(rep_positions) < 3:
        return {
            'overall_score': 0, 'consistency_score': 0,
            'smoothness_score': 0, 'range_of_motion_score': 0,
            'rep_quality_scores': [], 'num_cycles': 0,
            'avg_distance': 0, 'feedback': '数据太短或未检测到足够的重复动作',
        }

    peaks_arr = np.array(rep_positions)

    # 提取并重采样每个周期
    cycles = []
    for i in range(len(peaks_arr) - 1):
        cycle = extract_cycle_from_peaks(data, peaks_arr, i)
        resampled = resample_cycle(cycle, target_len=200)
        if resampled is not None:
            cycles.append(resampled)

    if len(cycles) < 2:
        return {
            'overall_score': 50, 'consistency_score': 50,
            'smoothness_score': 50, 'range_of_motion_score': 50,
            'rep_quality_scores': [], 'num_cycles': len(cycles),
            'avg_distance': 0, 'feedback': '动作周期提取不足',
        }

    # 一致性评分
    template = np.mean(cycles, axis=0)
    distances = [float(np.sqrt(np.mean((c - template) ** 2))) for c in cycles]
    mean_dist = np.mean(distances)
    consistency_score = min(100, max(0, 100 - mean_dist * 200))

    # 平滑度评分
    if data.shape[1] >= 3:
        jerk = np.diff(data[:, :3], axis=0)
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
        smoothness_score = min(100, max(0, 100 - jerk_rms * 100))
    else:
        smoothness_score = 70

    # 活动度评分
    energy = compute_signal_energy(data)
    amplitude = float(np.std(energy))
    rom_score = min(100, max(0, amplitude * 80 + 30))

    # 每个 rep 评分
    rep_scores = [min(100, max(0, 100 - d * 200)) for d in distances]

    # 综合
    overall = round(0.4 * consistency_score + 0.3 * smoothness_score + 0.3 * rom_score, 1)

    # 反馈
    feedback = []
    if consistency_score < 60:
        feedback.append("动作一致性较低，建议保持节奏稳定")
    if smoothness_score < 60:
        feedback.append("动作流畅度偏低，减少急停急起")
    if rom_score < 50:
        feedback.append("动作幅度偏小，建议增大活动范围")
    if consistency_score >= 80 and smoothness_score >= 80:
        feedback.append("动作质量优秀！")

    return {
        'overall_score': overall,
        'consistency_score': round(consistency_score, 1),
        'smoothness_score': round(smoothness_score, 1),
        'range_of_motion_score': round(rom_score, 1),
        'rep_quality_scores': [round(s, 1) for s in rep_scores],
        'num_cycles': len(cycles),
        'avg_distance': round(mean_dist, 4),
        'feedback': ' | '.join(feedback) if feedback else '动作质量正常',
    }


def detect_anomaly(data, window_size=200):
    """基于统计阈值的异常动作检测。

    Returns:
        list[dict]: 异常片段 [{start, end, type, severity, detail}]
    """
    anomalies = []
    acc_magnitude = np.sqrt(np.sum(data[:, :3] ** 2, axis=1)) if data.shape[1] >= 3 else np.zeros(len(data))

    # 异常静止检测
    dead_threshold = 0.01
    dead_start = None
    for i in range(len(acc_magnitude)):
        if acc_magnitude[i] < dead_threshold and dead_start is None:
            dead_start = i
        elif acc_magnitude[i] >= dead_threshold and dead_start is not None:
            if i - dead_start > window_size:
                anomalies.append({
                    'start': dead_start, 'end': i, 'type': 'low_activity',
                    'severity': 'warning',
                    'detail': f'传感器静止 {i - dead_start} 帧，可能脱落',
                })
            dead_start = None

    # 能量尖峰检测
    energy = compute_signal_energy(data)
    global_std = np.std(energy)
    spike_threshold = np.mean(energy) + 5 * global_std
    spike_idx = np.where(energy > spike_threshold)[0]
    if len(spike_idx) > 0:
        clusters = np.split(spike_idx, np.where(np.diff(spike_idx) > 50)[0] + 1)
        for cluster in clusters:
            if len(cluster) > 10:
                anomalies.append({
                    'start': int(cluster[0]), 'end': int(cluster[-1]),
                    'type': 'energy_spike', 'severity': 'warning',
                    'detail': f'检测到异常高能信号 ({len(cluster)} 帧)',
                })

    return anomalies


def full_analysis(data, rep_counter=None, classifier=None, class_names=None):
    """综合分析：评分 + 分组 + 异常检测。

    Args:
        data: (N, C) 传感器数据
        rep_counter: RepCounter 实例（可选）
        classifier: ExerciseClassifier 实例（可选）
        class_names: 类别名称列表

    Returns:
        dict: 包含 exercise, sets, quality, anomalies 四个部分的完整分析
    """
    # Rep 检测
    if rep_counter is not None:
        reps = rep_counter.count_reps(data)
        rep_positions = reps.get('rep_positions', [])
    else:
        energy = compute_signal_energy(data)
        b, a = signal.butter(3, 0.1)
        energy_filtered = signal.filtfilt(b, a, energy)
        prominence = np.std(energy_filtered) * 0.5
        peaks, _ = signal.find_peaks(energy_filtered, distance=100, prominence=prominence)
        rep_positions = peaks.tolist()

    # 组数分割
    sets = segment_into_sets(rep_positions)

    # 质量评分
    quality = score_exercise_quality(data, rep_positions)

    # 异常检测
    anomalies = detect_anomaly(data)

    # 动作识别
    exercise = 'unknown'
    if classifier is not None:
        try:
            result = classifier.classify(data)
            exercise = result.get('prediction', 'unknown')
        except Exception:
            pass

    return {
        'exercise': exercise,
        'sets': sets,
        'quality': quality,
        'anomalies': anomalies,
        'rep_count': len(rep_positions),
    }


def print_analysis_report(result):
    """打印格式化的分析报告。"""
    print("\n" + "=" * 60)
    print(f"  运动分析报告 - {result['exercise']}")
    print("=" * 60)

    print(f"\n  [动作计数]")
    print(f"    总重复次数: {result['rep_count']}")

    print(f"\n  [组数识别]")
    for s in result['sets']:
        print(f"    第 {s['set_num']} 组: {s['reps']} reps")

    q = result['quality']
    print(f"\n  [质量评分] (满分100)")
    print(f"    综合评分: {q['overall_score']}")
    print(f"    一致性:   {q['consistency_score']}")
    print(f"    平滑度:   {q['smoothness_score']}")
    print(f"    活动度:   {q['range_of_motion_score']}")
    print(f"    反馈:     {q['feedback']}")

    if result['anomalies']:
        print(f"\n  [异常检测]")
        for a in result['anomalies']:
            print(f"    [{a['severity']}] {a['type']}: {a['detail']}")

    print("=" * 60)
