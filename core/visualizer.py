# -*- coding: utf-8 -*-
"""
可视化模块
========================================
画检测框、类别文字、半透明统计面板、计数线。
所有颜色方案统一在这里定义，改风格只动这一个文件。
"""
import cv2

# ---------- 颜色定义（BGR） ----------
GREEN = (0, 255, 0)        # 检测框
RED = (0, 0, 255)          # 总数文字 / 中心点
YELLOW = (0, 255, 255)     # 本帧瓶数
ORANGE = (255, 128, 0)     # 计数线 / 过线统计
WHITE = (255, 255, 255)    # 拖拽框等
GRAY = (200, 200, 200)     # 帧数提示

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(img, x1, y1, x2, y2, color=GREEN, thickness=2):
    """画单个检测框"""
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_label(img, text, x, y, color=GREEN, scale=0.5, thickness=2):
    """在框上方画类别 + 置信度文字"""
    cv2.putText(img, text, (int(x), int(y) - 8),
                FONT, scale, color, thickness, cv2.LINE_AA)


def draw_line(img, x1, y1, x2, y2, color=ORANGE, thickness=2):
    """画线（计数线等）"""
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_total(img, text, color=RED, scale=1.2, thickness=3):
    """左上角画总数"""
    cv2.putText(img, text, (20, 40), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img, lines, origin=(16, 16), width=460, line_h=42, pad=18, alpha=0.78):
    """在左上角铺一块半透明黑底面板，逐行写字

    参数：
        lines : [(文字, 颜色), ...]，每行一条
        alpha : 黑底不透明度（0.78 压得比较实，避免底下检测框小字透上来）
    """
    x, y = origin
    height = pad * 2 + line_h * len(lines)

    # 先画实心矩形，再与原图按比例混合 = 半透明效果
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x + pad, y + pad + line_h * (i + 1) - 12),
                    FONT, 0.85, color, 2, cv2.LINE_AA)
    return img
