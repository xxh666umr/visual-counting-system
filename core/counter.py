# -*- coding: utf-8 -*-
"""
计数模块
========================================
ImageCounter  —— 图片 / 文件夹计数：检测 → 筛选 → 计数 → 可视化 → 保存
VideoCounter  —— 视频逐帧计数：检测 + 跟踪去重 + 过线统计 + CSV/JSON 导出

计数规则（与项目说明一致）：
1. 只数目标类别（默认 bottle），模型只算这一类 + 每个框再回查类别名，双保险
2. 置信度阈值 conf = 0.15：实测 0.4 漏检多、0.05 误检多，0.15 是折中点
3. 视频里同一个瓶子会被逐帧重复计数，用 ByteTrack 跟踪 ID 去重（unique）
4. 过线计数：按每个 ID 中心点 y 与计数线（画面 60% 高度处）的位置关系判断
"""
import os

import cv2

from . import io_utils
from . import visualizer as viz


# ==================== 图片计数 ====================
class ImageCounter:
    """图片 / 文件夹计数"""

    def __init__(self, config, detector):
        self.config = config
        self.detector = detector

    def count_image(self, img_path: str) -> dict:
        """单张图片计数：返回 {'file', 'count', 'target'}，并把结果图存到 output/"""
        img = io_utils.imread_cn(img_path)
        if img is None:
            print(f"[跳过] 无法读取图片：{img_path}")
            return {}

        results = self.detector.predict(img)
        boxes = results.boxes
        count = 0

        for box in boxes:
            cls_name = self.detector.names[int(box.cls[0])]
            if cls_name != self.config.target_class:   # 类别双保险
                continue
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            viz.draw_box(img, x1, y1, x2, y2)
            viz.draw_label(img, f"{cls_name} {conf:.2f}", x1, y1)

        viz.draw_total(img, f"Total: {count}")

        # 保存结果图（中文路径兼容）
        io_utils.ensure_dir(self.config.output_dir)
        out_path = os.path.join(self.config.output_dir, os.path.basename(img_path))
        io_utils.imwrite_cn(out_path, img)

        result = {
            "file": os.path.basename(img_path),
            "count": count,
            "target": self.config.target_class,
        }
        print(f"{result['file']}  检测到 {count} 个 {self.config.target_class}")
        return result

    def process_folder(self, folder: str) -> list:
        """批量处理一个文件夹里的所有图片，返回结果列表"""
        files = io_utils.list_images(folder)
        if not files:
            print(f"[提示] 文件夹里没有图片：{folder}")
            return []

        results = []
        for path in files:
            r = self.count_image(path)
            if r:
                results.append(r)

        total = sum(r["count"] for r in results)
        print(f"\n文件夹合计：{len(results)} 张图，共 {total} 个 {self.config.target_class}")
        return results


# ==================== 视频计数 ====================
class VideoCounter:
    """视频计数：逐帧检测 + 跟踪去重 + 过线统计"""

    def __init__(self, config, detector):
        self.config = config
        self.detector = detector

    def count_video(self, video_path: str, show_window: bool = False) -> dict:
        """处理一段视频，导出结果视频 + CSV + JSON 摘要"""
        if not os.path.exists(video_path):
            print(f"[跳过] 视频不存在：{video_path}")
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[跳过] 打不开视频：{video_path}")
            return {}

        # 读取视频元信息
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        line_y = int(h * self.config.line_ratio)     # 计数线 y 坐标

        name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = io_utils.ensure_dir(self.config.output_dir)
        out_mp4 = os.path.join(out_dir, f"{name}_count.mp4")
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 换视频得让跟踪器从零开始
        self.detector.reset_trackers()

        # 统计状态
        seen = set()        # 出现过的跟踪 ID（自动去重 → unique 数）
        frame_sum = 0       # 逐帧瓶数累加（会重复计数，仅作对比）
        last_cy = {}        # 每个 ID 上一帧的中心点 y，判断过线
        down = 0            # 从上往下穿线
        up = 0              # 从下往上穿线
        rows = []           # 逐帧 CSV 数据
        frame_no = 0

        print(f"\n视频 {os.path.basename(video_path)}：{w}x{h}，{fps:.0f} fps，共 {total_frames} 帧")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1

            # 用 track 代替 predict：persist=True 记住上一帧，同一瓶子保持同一 ID
            results = self.detector.track(frame)
            ids = results.boxes.id
            if ids is None:                      # 开头几帧还没分配 ID
                ids = [-1] * len(results.boxes)

            n_now = 0
            for box, tid in zip(results.boxes, ids):
                cls_name = self.detector.names[int(box.cls[0])]
                if cls_name != self.config.target_class:   # 类别筛选
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                n_now += 1

                tid = int(tid)
                if tid >= 0:
                    seen.add(tid)
                    # 过线判断：上一帧在线的一边，这一帧到了另一边
                    if tid in last_cy:
                        if last_cy[tid] < line_y <= cy:
                            down += 1
                        elif last_cy[tid] >= line_y > cy:
                            up += 1
                    last_cy[tid] = cy

                # 画框 + 中心点（小框不写字，避免糊成一团）
                viz.draw_box(frame, x1, y1, x2, y2)
                if x2 - x1 >= self.config.min_label_w:
                    label = f"{cls_name} {conf:.2f}" if tid < 0 else f"id{tid} {cls_name} {conf:.2f}"
                    viz.draw_label(frame, label, x1, y1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            frame_sum += n_now

            # 计数线 + 半透明信息面板
            viz.draw_line(frame, 0, line_y, w, line_y)
            viz.draw_panel(frame, [
                (f"{self.config.target_class}(this frame): {n_now}", viz.YELLOW),
                (f"unique (tracked):  {len(seen)}", viz.GREEN),
                (f"line  down {down}  up {up}", (255, 180, 80)),
                (f"frame {frame_no}/{total_frames}", viz.GRAY),
            ])

            writer.write(frame)
            rows.append([frame_no, n_now, len(seen), down, up])

            if show_window:
                cv2.imshow("object count", frame)
                if cv2.waitKey(1) == 27:         # 按 Esc 提前退出
                    break

            if frame_no % 30 == 0:
                print(f"  第 {frame_no}/{total_frames} 帧：本帧 {n_now} 个，"
                      f"累计不同 {len(seen)} 个")

        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

        # ---------- 导出结构化结果 ----------
        csv_path = io_utils.save_csv(
            os.path.join(out_dir, f"{name}_counts.csv"),
            ["frame", "bottles_in_frame", "unique_tracked", "line_down", "line_up"],
            rows,
        )

        summary = {
            "video": os.path.basename(video_path),
            "target": self.config.target_class,
            "conf_threshold": self.config.conf_threshold,
            "total_frames": frame_no,
            "frame_sum": frame_sum,
            "unique_tracked": len(seen),
            "peak_in_frame": max((r[1] for r in rows), default=0),
            "avg_in_frame": round(frame_sum / len(rows), 2) if rows else 0,
            "line_down": down,
            "line_up": up,
            "output_video": out_mp4,
            "csv": csv_path,
        }
        json_path = io_utils.save_json(
            os.path.join(out_dir, f"{name}_summary.json"), summary)
        summary["json"] = json_path

        print(f"\n视频统计：")
        print(f"  逐帧累加（会重复计数）  : {frame_sum}")
        print(f"  跟踪去重后一共出现过    : {len(seen)} 个")
        print(f"  单帧最多                : {summary['peak_in_frame']} 个")
        print(f"  平均每帧                : {summary['avg_in_frame']} 个")
        print(f"  过线：向下 {down} 个，向上 {up} 个")
        print(f"  结果视频 → {out_mp4}")
        print(f"  逐帧数据 → {csv_path}")
        print(f"  统计摘要 → {json_path}")
        return summary
