# -*- coding: utf-8 -*-
"""
检测器模块
========================================
封装 YOLO 模型：
- 惰性加载：第一次使用时才加载模型，启动更快
- 自动解析目标类别 ID：支持命令行指定任意 COCO 类别名（bottle / car / person...）
- 提供 predict（图片检测）与 track（视频跟踪）两种接口
"""
import os
from typing import Optional

from ultralytics import YOLO


class Detector:
    """YOLO 检测器封装"""

    def __init__(self, config):
        self.config = config
        self._model: Optional[YOLO] = None
        self._target_id: Optional[int] = None

    # ==================== 模型加载 ====================
    @property
    def model(self) -> YOLO:
        """惰性加载：第一次访问时才真正读模型文件"""
        if self._model is None:
            self._load()
        return self._model

    def _load(self):
        path = self.config.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"模型文件不存在：{path}\n"
                f"请把 yolov8n.pt 放到项目根目录，或用 --model 指定其他权重。"
            )
        self._model = YOLO(path)

    @property
    def names(self) -> dict:
        """模型能识别的全部类别：{ID: 类别名}"""
        return self.model.names

    # ==================== 目标类别解析 ====================
    @property
    def target_id(self) -> int:
        """把目标类别名解析成类别 ID（不存在则明确报错）"""
        if self._target_id is None:
            for idx, name in self.names.items():
                if name == self.config.target_class:
                    self._target_id = idx
                    break
            if self._target_id is None:
                raise ValueError(
                    f"模型里没有类别 '{self.config.target_class}'。\n"
                    f"可用类别：{list(self.names.values())}"
                )
        return self._target_id

    # ==================== 推理接口 ====================
    def predict(self, image, **kwargs):
        """单张图片检测（只检测目标类别，速度更快）"""
        return self.model.predict(
            image,
            conf=self.config.conf_threshold,
            imgsz=self.config.imgsz,
            classes=[self.target_id],
            verbose=False,
            **kwargs,
        )[0]

    def track(self, frame, **kwargs):
        """视频帧跟踪（persist=True 记住上一帧，同一目标保持同一 ID）"""
        return self.model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            classes=[self.target_id],
            conf=self.config.conf_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
            **kwargs,
        )[0]

    def reset_trackers(self):
        """换视频时重置跟踪器，避免上一段的 ID 串到这一段来"""
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
