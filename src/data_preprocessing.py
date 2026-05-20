"""
数据预处理入口
================================================================
支持多数据集预处理，输出统一的 .npy 格式。

可用数据集:
  python src/data_preprocessing.py                    # 默认 gym_gesture
  python src/data_preprocessing.py --dataset mmfit    # MM-Fit
  python src/data_preprocessing.py --dataset gym_gesture --dataset mmfit  # 全部
  python src/data_preprocessing.py --dataset all      # 全部数据集

单独运行:
  python src/datasets/preprocess_gym_gesture.py
  python src/datasets/preprocess_mmfit.py
  python src/datasets/preprocess_recofit.py

多数据集合并:
  python src/datasets/merge_datasets.py --datasets gym_gesture mmfit recofit
"""

import argparse

DATASETS = {
    'gym_gesture': 'datasets.preprocess_gym_gesture',
    'mmfit': 'datasets.preprocess_mmfit',
    'recofit': 'datasets.preprocess_recofit',
}


def main():
    parser = argparse.ArgumentParser(description='多数据集预处理入口')
    parser.add_argument('--dataset', nargs='+', default=['gym_gesture'],
                        choices=['gym_gesture', 'mmfit', 'recofit', 'all'],
                        help='要预处理的数据集 (默认: gym_gesture)')
    parser.add_argument('--merge', action='store_true',
                        help='预处理后自动合并所有数据集')
    args = parser.parse_args()

    selected = args.dataset
    if 'all' in selected:
        selected = list(DATASETS.keys())

    for ds_name in selected:
        if ds_name not in DATASETS:
            print(f"未知数据集: {ds_name}")
            continue

        module_name = DATASETS[ds_name]
        try:
            mod = __import__(module_name, fromlist=['preprocess'])
        except ImportError as e:
            print(f"导入 {module_name} 失败: {e}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  处理: {ds_name}")
        print(f"{'=' * 60}")
        mod.preprocess()

    if args.merge:
        print(f"\n{'=' * 60}")
        print("  合并所有可用数据集")
        print(f"{'=' * 60}")
        from datasets.merge_datasets import merge_datasets
        merge_datasets(selected)


if __name__ == '__main__':
    main()
