"""
数据预处理入口（向后兼容）
================================================================
默认处理 Gym Gesture 数据集。

如需处理其他数据集:
  python src/datasets/preprocess_recofit.py
  python src/datasets/preprocess_recgym.py

多数据集合并:
  python src/datasets/merge_datasets.py --datasets gym_gesture recofit recgym

查看所有可用数据集:
  python scripts/download_datasets.py
"""

from datasets.preprocess_gym_gesture import preprocess

if __name__ == '__main__':
    preprocess()
