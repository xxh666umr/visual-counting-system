# -*- coding: utf-8 -*-
"""
视觉智能计数系统 —— 数瓶子（优化版入口）
========================================
支持三种输入：一张图片 / 一个图片文件夹 / 一段视频 .mp4

用法：
    python main.py                                  # 默认：跑 data/images 和 data/videos
    python main.py 图片路径.jpg                     # 只跑一张图
    python main.py data/images                      # 只跑一个文件夹
    python main.py data/videos/xx.mp4 --show        # 跑视频并实时弹窗预览
    python main.py --target car --conf 0.25         # 数别的类别 / 调阈值

输出（都在 output/ 下）：
    <图片名>.jpg              带框结果图
    <视频名>_count.mp4        带实时面板的结果视频
    <视频名>_counts.csv       逐帧数据（可画曲线）
    <视频名>_summary.json     统计摘要
"""
import argparse
import os
import sys

from config import Config
from core import io_utils
from core.counter import ImageCounter, VideoCounter
from core.detector import Detector


def build_parser() -> argparse.ArgumentParser:
    """命令行参数解析（标准 argparse，替代手工解析 sys.argv）"""
    parser = argparse.ArgumentParser(
        description="视觉智能计数系统：用 YOLO 数图片 / 文件夹 / 视频里的目标",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", help="输入：图片 / 文件夹 / 视频（可多个）")
    parser.add_argument("--show", action="store_true", help="跑视频时实时弹窗预览")
    parser.add_argument("--target", default="bottle", help="要数的目标类别（COCO 80 类名）")
    parser.add_argument("--conf", type=float, default=0.15, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--model", default="", help="模型权重路径（默认项目根目录 yolov8n.pt）")
    parser.add_argument("--output", default="", help="输出目录（默认 output/）")
    return parser


def main():
    args = build_parser().parse_args()

    # 组装配置：命令行参数覆盖默认值
    config = Config(
        target_class=args.target,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        model_path=args.model,
        output_dir=args.output,
    )

    # 提前加载模型，模型 / 类别不存在时立即报错，而不是跑到一半才失败
    detector = Detector(config)
    try:
        detector.model
        print(f"模型已加载：{config.model_path}")
        print(f"目标类别：{config.target_class}（ID={detector.target_id}）")
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)

    img_counter = ImageCounter(config, detector)
    vid_counter = VideoCounter(config, detector)

    # ---------- 1) 指定了输入 → 逐一分发 ----------
    if args.paths:
        for path in args.paths:
            if os.path.isdir(path):
                img_counter.process_folder(path)
            elif path.lower().endswith(io_utils.VIDEO_EXTS):
                vid_counter.count_video(path, show_window=args.show)
            else:
                img_counter.count_image(path)
        return

    # ---------- 2) 没指定输入 → 跑默认全套（图片 → 视频） ----------
    print("=" * 50)
    print(f"1) 图片计数：{config.images_dir}")
    print("=" * 50)
    img_counter.process_folder(config.images_dir)

    videos = io_utils.list_videos(config.videos_dir)
    if not videos:
        print("\ndata/videos 里没有视频。放一段 .mp4 进去，"
              "或运行 scripts/make_demo_video.py 生成演示视频。")
        return

    print("\n" + "=" * 50)
    print("2) 视频计数：data/videos")
    print("=" * 50)
    for v in videos:
        vid_counter.count_video(v, show_window=args.show)


if __name__ == "__main__":
    main()
