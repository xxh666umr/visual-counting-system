# -*- coding: utf-8 -*-
"""
路径 B · 用标注好的数据微调 YOLOv8n（CPU 版）
========================================
默认用 datasets/pills 药片数据集（92 训练 + 23 验证，实验包自带，已标注）；
换自己的数据：运行时输入 data（里面要有 images/train、images/val 及对应
labels/train、labels/val，还有 classes.txt）。

用法：
    python scripts/train_custom.py

训练完成后权重在 runs/train/weights/best.pt，
用它跑计数：python main.py --model runs/train/weights/best.pt
"""
import os

import yaml
from ultralytics import YOLO

# scripts/ 的上一级 = 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    folder = input("数据集文件夹（回车 = datasets/pills，自己的数据输入 data）：").strip()
    if not folder:
        folder = os.path.join(BASE_DIR, "datasets", "pills")
    data_dir = os.path.abspath(folder)
    if not os.path.isdir(os.path.join(data_dir, "images")):
        print("找不到", os.path.join(data_dir, "images"))
        return

    # 类别名：classes.txt（本工具生成）或 labels.txt（makesense.ai 导出）
    cls_file = os.path.join(data_dir, "classes.txt")
    if not os.path.exists(cls_file):
        cls_file = os.path.join(data_dir, "labels.txt")
    names = [l.strip() for l in open(cls_file, encoding="utf-8") if l.strip()]

    # 训练与验证必须分开，避免在训练集上评估
    if all(os.path.isdir(os.path.join(data_dir, kind, split))
           for kind in ("images", "labels") for split in ("train", "val")):
        train, val = "images/train", "images/val"
    else:
        raise ValueError("请先划分 images/train、images/val 及对应 labels/train、labels/val；"
                         "不能用同一批图训练和验证。")

    # 生成 data.yaml 数据集配置
    cfg = {"path": data_dir, "train": train, "val": val,
           "names": {i: n for i, n in enumerate(names)}}
    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print("类别：", cfg["names"], "  训练集：", train, "  验证集：", val)

    # 微调训练（CPU）
    model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
    model.train(data=yaml_path, epochs=30, imgsz=640, batch=8,
                device="cpu", workers=0, plots=False,
                project=os.path.join(BASE_DIR, "runs"), name="train", exist_ok=True)
    print("训练完成，权重：runs/train/weights/best.pt")


if __name__ == "__main__":
    main()
