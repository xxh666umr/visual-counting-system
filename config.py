# -*- coding: utf-8 -*-
"""
全局配置模块
========================================
所有可调参数集中在这里统一管理，命令行参数（--conf / --target 等）会覆盖默认值。

修改参数建议：
- 临时调试  → 用命令行参数（如 python main.py --conf 0.25）
- 长期修改  → 直接改下面的默认值
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """系统配置（dataclass，所有参数一目了然）"""

    # ---------- 项目根目录（基于本文件位置自动推导，跨机器可移植） ----------
    base_dir: str = os.path.dirname(os.path.abspath(__file__))

    # ---------- 检测参数 ----------
    model_path: str = ""            # 模型权重路径（空 = 项目根目录 yolov8n.pt）
    target_class: str = "bottle"    # 要数的目标类别（COCO 80 类之一，见任务书附录）
    conf_threshold: float = 0.15    # 置信度阈值：实测 0.4 漏检多、0.05 误检多，0.15 是折中点
    imgsz: int = 640                # 输入图片尺寸（越大越准越慢）

    # ---------- 视频参数 ----------
    tracker: str = "bytetrack.yaml"     # 跟踪器：让同一个目标保持同一 ID，用于去重
    line_ratio: float = 0.60            # 计数线位置：画面高度的 60% 处
    min_label_w: int = 70               # 框宽小于此像素就不写文字，避免糊成一团

    # ---------- 目录参数 ----------
    data_dir: str = ""                  # 数据目录（空 = base_dir/data）
    output_dir: str = ""                # 输出目录（空 = base_dir/output）

    # ---------- 派生路径（自动生成，不用手填） ----------
    images_dir: str = ""
    videos_dir: str = ""

    def __post_init__(self):
        """把依赖 base_dir 的目录参数补全"""
        if not self.model_path:
            local = os.path.join(self.base_dir, "yolov8n.pt")
            # 本地没有权重时退回裸文件名，ultralytics 会自动下载官方 yolov8n.pt
            self.model_path = local if os.path.isfile(local) else "yolov8n.pt"
        if not self.data_dir:
            self.data_dir = os.path.join(self.base_dir, "data")
        if not self.output_dir:
            self.output_dir = os.path.join(self.base_dir, "output")
        self.images_dir = os.path.join(self.data_dir, "images")
        # data/images 不存在时退回仓库自带的 samples/，保证 clone 下来就能跑
        if not os.path.isdir(self.images_dir):
            samples = os.path.join(self.base_dir, "samples")
            if os.path.isdir(samples):
                self.images_dir = samples
        self.videos_dir = os.path.join(self.data_dir, "videos")
