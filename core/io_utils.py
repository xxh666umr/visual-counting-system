# -*- coding: utf-8 -*-
"""
IO 工具模块
========================================
1. 中文路径兼容的图片读写：
   Windows 上 cv2.imread / cv2.imwrite 按系统编码找文件，
   路径一带中文（如"Python视觉AI实验\项目一"）就会失败，
   这里用 numpy 中转字节流解决。
2. CSV / JSON 结构化导出。
3. 目录自动创建、文件列表收集。
"""
import csv
import json
import os

import cv2
import numpy as np

# 支持的扩展名（统一在这里定义）
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def ensure_dir(path: str) -> str:
    """确保目录存在，返回该路径"""
    os.makedirs(path, exist_ok=True)
    return path


def imread_cn(path: str):
    """读取图片（兼容中文路径），失败返回 None"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_cn(path: str, img) -> bool:
    """写入图片（兼容中文路径），成功返回 True"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def save_csv(path: str, header, rows) -> str:
    """保存 CSV（utf-8-sig 编码，Excel 打开不乱码），返回路径"""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def save_json(path: str, data) -> str:
    """保存 JSON 摘要（UTF-8），返回路径"""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def list_images(folder: str):
    """列出文件夹里所有图片（按文件名排序）"""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMAGE_EXTS)]


def list_videos(folder: str):
    """列出文件夹里所有视频（按文件名排序）"""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(VIDEO_EXTS)]
