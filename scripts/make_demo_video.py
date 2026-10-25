# -*- coding: utf-8 -*-
"""
生成演示视频（给视频计数功能用的素材）
========================================
把 data/images 里的瓶子照片，做成一段"手机随手拍"的 mp4：
每张照片 = 一个镜头，缓慢推近 / 拉远 / 左右平移（Ken Burns 运镜），
再叠一点手持抖动 + 颗粒 + 旋转，看起来像边走边拍。

用法：
    python scripts/make_demo_video.py

输出：data/videos/bottle_walk.mp4
"""
import math
import os

import cv2
import numpy as np

# scripts/ 的上一级 = 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "data", "images")
OUT_DIR = os.path.join(BASE_DIR, "data", "videos")
OUT_MP4 = os.path.join(OUT_DIR, "bottle_walk.mp4")

W, H, FPS = 1280, 720, 30      # 输出画布：720p，30 帧/秒
SHOT_SEC = 2.0                 # 每个镜头时长（秒）
COVER = 1.45                   # 照片放大到画布的多少倍，留出平移余量

# 镜头顺序：文件名 + 运镜方式。瓶子数量从小排到大（5 / 18 / 14 / 42 / 62），
# 画面上"实时瓶数"一直在变；中间那段用上下摇，瓶子会穿过计数线，
# 过线计数那两个数字才有东西可看。
SHOTS = [
    ("img08.jpg", "push"),        # 5 个
    ("img01.jpg", "pan_lr"),      # 18 个
    ("img09.jpg", "tilt_down"),   # 14 个
    ("img13.jpg", "push"),        # 42 个
    ("img10.jpg", "pull"),        # 62 个（仓库货架，收尾）
]


def ease(t: float) -> float:
    """平滑缓动，镜头起停不生硬"""
    return t * t * (3.0 - 2.0 * t)


def motion(kind: str, t: float):
    """返回该帧的 (放大倍数 z, 中心横向比例 cxr, 中心纵向比例 cyr)"""
    e = ease(t)
    if kind == "push":                       # 缓慢推近
        return 1.03 + 0.20 * e, 0.5, 0.5
    if kind == "pull":                       # 缓慢拉远
        return 1.25 - 0.20 * e, 0.5, 0.5
    if kind == "pan_lr":                     # 从左摇到右
        return 1.14 + 0.04 * math.sin(math.pi * e), 0.10 + 0.80 * e, 0.5
    if kind == "pan_rl":                     # 从右摇到左
        return 1.14 + 0.04 * math.sin(math.pi * e), 0.90 - 0.80 * e, 0.5
    if kind == "diag":                       # 斜着走
        return 1.06 + 0.14 * e, 0.20 + 0.60 * e, 0.78 - 0.48 * e
    if kind == "tilt_up":                    # 镜头往上抬
        return 1.10 + 0.05 * e, 0.5, 0.85 - 0.70 * e
    if kind == "tilt_down":                  # 镜头往下压
        return 1.10 + 0.05 * e, 0.5, 0.15 + 0.70 * e
    return 1.0, 0.5, 0.5


def imread_cn(path: str):
    """读取图片（兼容中文路径）"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_scaled(path: str):
    """把照片放大到宽 >= W*COVER 且 高 >= H*COVER，给运镜留余量"""
    img = imread_cn(path)
    if img is None:
        return None
    k = max(W * COVER / img.shape[1], H * COVER / img.shape[0])
    return cv2.resize(img, (int(round(img.shape[1] * k)), int(round(img.shape[0] * k))),
                      interpolation=cv2.INTER_AREA)


def render_shot(big, kind: str, n_frames: int, seed: float):
    """渲染一个镜头，逐帧 yield"""
    SH, SW = big.shape[:2]

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)         # 0 → 1
        z, cxr, cyr = motion(kind, t)

        # 裁剪窗口大小：放得越大，窗口越小
        cw, ch = int(W / z), int(H / z)

        # 手持抖动：两个不同频率的正弦叠加，才不像机器在走
        sx = 5.0 * math.sin(2 * math.pi * 1.9 * t + seed) \
             + 8.0 * math.sin(2 * math.pi * 0.55 * t + seed * 1.7)
        sy = 6.0 * math.cos(2 * math.pi * 1.4 * t + seed * 1.3) \
             + 7.0 * math.cos(2 * math.pi * 0.42 * t + seed * 2.1)

        # 窗口中心最终位置，夹边界别裁到照片外
        cx = cw / 2 + cxr * (SW - cw) + sx
        cy = ch / 2 + cyr * (SH - ch) + sy
        cx = min(max(cx, cw / 2), SW - cw / 2)
        cy = min(max(cy, ch / 2), SH - ch / 2)

        x1, y1 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
        crop = big[y1:y1 + ch, x1:x1 + cw]
        frame = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

        # 轻微手持旋转（±0.4 度）
        ang = 0.40 * math.sin(2 * math.pi * 0.75 * t + seed)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        frame = cv2.warpAffine(frame, M, (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 曝光呼吸：像自动曝光在微调
        alpha = 1.0 + 0.015 * math.sin(2 * math.pi * 0.6 * t + seed * 0.9)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

        # 画面颗粒：加一点噪声才像真拍的
        noise = np.random.randint(-3, 4, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        yield frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    writer = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        print("打不开视频写入器，检查一下 OpenCV 的 mp4v 编码器")
        return

    n_frames = int(SHOT_SEC * FPS)
    total = 0

    for idx, (fname, kind) in enumerate(SHOTS):
        path = os.path.join(SRC_DIR, fname)
        if not os.path.exists(path):
            print(f"跳过（照片不存在）：{fname}")
            continue

        big = load_scaled(path)
        if big is None:
            print(f"跳过（读不出来）：{fname}")
            continue

        seed = idx * 1.618     # 每段抖动相位不同，避免抖得一模一样
        for frame in render_shot(big, kind, n_frames, seed):
            writer.write(frame)
            total += 1
        print(f"[{idx + 1}/{len(SHOTS)}] {fname}  运镜={kind}  已渲染 {total} 帧")

    writer.release()
    print(f"\n搞定：{OUT_MP4}")
    print(f"时长约 {total / FPS:.1f} 秒，{total} 帧，{W}x{H} @ {FPS}fps")
    print("接下来跑 python main.py，它会读这段视频，逐帧显示瓶子个数")


if __name__ == "__main__":
    main()
