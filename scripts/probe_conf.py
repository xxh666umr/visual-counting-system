# -*- coding: utf-8 -*-
"""
阈值探索工具
========================================
对比同一批图片在 0.40 / 0.25 / 0.15 三个置信度阈值下的检出数量，
找出置信度偏低的目标（0.15 能检出、0.40 检不出的），
帮你判断"阈值该定多少" —— 项目里 0.15 就是这么试出来的。

用法：
    python scripts/probe_conf.py [图片文件夹]
"""
import os
import sys

from ultralytics import YOLO

# scripts/ 的上一级 = 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)          # 让 import config 生效

from config import Config              # noqa: E402
from core import io_utils              # noqa: E402

TARGET = "bottle"
THRESHOLDS = (0.40, 0.25, 0.15)


def main():
    folder = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not folder:
        folder = os.path.join(BASE_DIR, "data", "images")
    files = io_utils.list_images(folder)
    if not files:
        print(f"文件夹里没有图片：{folder}")
        return

    model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
    target_id = None
    for idx, name in model.names.items():
        if name == TARGET:
            target_id = idx
            break
    if target_id is None:
        print(f"模型里没有类别 {TARGET}")
        return

    print(f"{'file':12s} {'c0.40':>6s} {'c0.25':>6s} {'c0.15':>6s}  低置信瓶(<0.4)")
    tot = {t: 0 for t in THRESHOLDS}
    for f in files:
        img = io_utils.imread_cn(f)
        if img is None:
            continue
        counts, low = {}, []
        for thr in THRESHOLDS:
            r = model(img, conf=thr, imgsz=640, classes=[target_id], verbose=False)[0]
            counts[thr] = len(r.boxes)
            tot[thr] += len(r.boxes)
        # 找出 0.15 能检出、0.40 检不出的瓶子（置信度 0.15~0.40 之间）
        r15 = model(img, conf=0.15, imgsz=640, classes=[target_id], verbose=False)[0]
        for box in r15.boxes:
            cf = float(box.conf[0])
            if cf < 0.40:
                low.append(round(cf, 3))
        print(f"{os.path.basename(f):12s} {counts[0.40]:6d} {counts[0.25]:6d} "
              f"{counts[0.15]:6d}  {sorted(low)}")

    print("\nTOTAL  0.40:", tot[0.40], " 0.25:", tot[0.25], " 0.15:", tot[0.15])
    print("0.4 相比 0.15 漏掉总数:", tot[0.15] - tot[0.40])


if __name__ == "__main__":
    main()
