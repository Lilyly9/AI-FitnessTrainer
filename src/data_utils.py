"""
数据配置工具 — 自动从 dataset_meta.json 读取类别信息。
所有脚本通过此模块获取 NUM_CLASSES / CLASS_NAMES，避免硬编码。
"""

import json
import os

DATA_DIR = 'data/processed/'
META_PATH = os.path.join(DATA_DIR, 'dataset_meta.json')

_DEFAULT_CLASS_NAMES = ['chest_fly', 'chest_press', 'lat_pulldown',
                        'seated_row', 'tricep_extension']


def load_meta():
    """加载合并后的数据集元信息。如果不存在则返回默认值。"""
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'num_classes': 5,
        'class_names': _DEFAULT_CLASS_NAMES,
    }


def get_num_classes():
    return load_meta()['num_classes']


def get_class_names():
    return load_meta()['class_names']


def get_label_sets():
    """返回各数据集的标签集合 {'gym_gesture': [0,1,2,3,4], 'recofit': [...], ...}。"""
    meta = load_meta()
    # dataset_meta.json 可能是 'label_sets' 或 'label_offsets' 格式
    if 'label_sets' in meta:
        return meta['label_sets']
    if 'label_offsets' in meta:
        offsets = meta['label_offsets']
        return {k: list(range(v[0], v[1])) for k, v in offsets.items()}
    return {}


# 模块级常量（首次 import 时读取）
_meta = load_meta()
NUM_CLASSES = _meta['num_classes']
CLASS_NAMES = _meta['class_names']
METADATA = _meta  # 完整元数据（供计算 per-dataset 指标）
