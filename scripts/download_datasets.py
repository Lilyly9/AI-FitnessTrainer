"""
数据集下载指引
================================================================
由于网络限制，数据集需手动下载。运行此脚本查看下载链接和放置位置。
"""

import os

DATASETS = {
    "gym_gesture": {
        "name": "Gym Gesture Classification (IEEE DataPort)",
        "url": "https://ieee-dataport.org/documents/gym-gesture-classification-using-imu-sensor-dataset",
        "files": ["imu_dataset.csv"],
        "target_dir": "data/raw/gym_gesture/",
        "note": "需注册 IEEE DataPort 账号（免费）",
    },
    "recofit": {
        "name": "Microsoft RecoFit (200+ participants)",
        "url": "https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors",
        "files": ["*.mat"],
        "target_dir": "data/raw/recofit/",
        "note": "git clone 或 Download ZIP，.mat 文件需 scipy 读取",
    },
    "recgym": {
        "name": "RecGym (UCI, 10人×12类, 2025)",
        "url": "https://archive.ics.uci.edu/dataset/1128",
        "files": ["*.csv"],
        "target_dir": "data/raw/recgym/",
        "note": "下载 ZIP 解压即可",
    },
    "mmfit": {
        "name": "MM-Fit (IMU+视频, 20 sessions)",
        "url": "https://zenodo.org/records/7672767",
        "files": ["*.csv"],
        "target_dir": "data/raw/mmfit/",
        "note": "多模态数据集，含同步视频",
    },
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("  可用的健身动作数据集")
    print("=" * 60)
    print()

    for key, ds in DATASETS.items():
        target = os.path.join(PROJECT_ROOT, ds["target_dir"])
        exists = os.path.exists(target) and any(
            f.endswith(('.csv', '.mat', '.zip')) for f in os.listdir(target)
        ) if os.path.exists(target) else False
        status = "[已下载]" if exists else "[需下载]"

        print(f"  {ds['name']}  {status}")
        print(f"    链接: {ds['url']}")
        print(f"    文件: {ds['files']}")
        print(f"    放入: {ds['target_dir']}")
        print(f"    备注: {ds['note']}")
        print()

    print("=" * 60)
    print("  使用流程")
    print("=" * 60)
    print("  1. 下载数据集放入对应的 data/raw/ 子目录")
    print("  2. 运行预处理:")
    for key in DATASETS:
        if key == "gym_gesture":
            print(f"     python src/datasets/preprocess_{key}.py  (已有)")
        else:
            print(f"     python src/datasets/preprocess_{key}.py")
    print("  3. 合并数据集（可选）:")
    print("     python src/datasets/merge_datasets.py --datasets gym_gesture recofit recgym")
    print("  4. 训练:")
    print("     python src/train.py")
    print()


if __name__ == '__main__':
    main()
