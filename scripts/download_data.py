"""下载并归一化 AISafetyLab 数据集到本地 data/ 目录。

用法（在项目根目录下）：
    python scripts/download_data.py
    python scripts/download_data.py --data_dir data
"""
from __future__ import annotations

import argparse
import os
import sys

# 确保能 import safellm 包（从项目根目录运行脚本时）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safellm.data.download import download_all


def main():
    p = argparse.ArgumentParser(description="下载并归一化 AISafetyLab 数据集到本地")
    p.add_argument("--data_dir", default="data", help="数据根目录")
    args = p.parse_args()
    download_all(args.data_dir)


if __name__ == "__main__":
    main()
