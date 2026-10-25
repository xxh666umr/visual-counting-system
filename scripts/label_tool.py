# -*- coding: utf-8 -*-
"""
离线标注小工具（OpenCV 实现）· 输出 YOLO 格式
========================================
用法：python scripts/label_tool.py [图片文件夹]
      （不填参数默认 data/images，标注自动存到 data/labels/，类别名存 classes.txt）

操作：
    鼠标拖动画框  |  数字键 0-9 切换类别  |  d 删除最后一个框
    空格 / n 下一张  |  p 上一张  |  q / Esc 退出
"""
import os
import sys

import cv2

# scripts/ 的上一级 = 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMG_DIR = os.path.join(BASE_DIR, "data", "images")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255)]
MAX_SIDE = 960

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_labels(txt: str, w: int, h: int):
    """读取 YOLO 格式标注（归一化 → 像素坐标）"""
    boxes = []
    if os.path.exists(txt):
        for line in open(txt, encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            boxes.append([c, int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                          int((cx + bw / 2) * w), int((cy + bh / 2) * h)])
    return boxes


def save_labels(txt: str, boxes, w: int, h: int):
    """保存 YOLO 格式标注（像素坐标 → 归一化）"""
    with open(txt, "w", encoding="utf-8") as f:
        for c, x1, y1, x2, y2 in boxes:
            f.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{abs(x2 - x1) / w:.6f} {abs(y2 - y1) / h:.6f}\n")


def main():
    img_dir = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or DEFAULT_IMG_DIR
    if not os.path.isdir(img_dir):
        print("找不到文件夹：", img_dir)
        input("按回车退出")
        return

    # 标注目录 = 图片目录的同级 labels 文件夹
    lbl_dir = os.path.join(os.path.dirname(os.path.abspath(img_dir)), "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    cls_file = os.path.join(os.path.dirname(os.path.abspath(img_dir)), "classes.txt")

    # 类别名：已有就读入，没有就交互创建
    if os.path.exists(cls_file):
        classes = [l.strip() for l in open(cls_file, encoding="utf-8") if l.strip()]
    else:
        s = input("输入类别名（英文，多个用逗号分开，如 bottle,cup）：").strip() or "object"
        classes = [c.strip() for c in s.split(",") if c.strip()]
        open(cls_file, "w", encoding="utf-8").write("\n".join(classes) + "\n")
    print("类别：", {i: c for i, c in enumerate(classes)}, "→ 已写入", cls_file)

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
    if not files:
        print("文件夹里没有图片")
        input("按回车退出")
        return
    print(f"共 {len(files)} 张图。鼠标拖动画框，数字键切类别，d 删框，空格下一张，p 上一张，q 退出")

    state = {"drag": None, "cur": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
            state["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            if abs(x - x0) > 4 and abs(y - y0) > 4:
                sc = param["scale"]
                param["boxes"].append([param["cls"], int(min(x0, x) / sc), int(min(y0, y) / sc),
                                       int(max(x0, x) / sc), int(max(y0, y) / sc)])
            state["drag"] = None

    cv2.namedWindow("Label Tool")
    i, cls_id = 0, 0
    while 0 <= i < len(files):
        path = os.path.join(img_dir, files[i])
        img = cv2.imread(path)
        if img is None:
            i += 1
            continue
        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        txt = os.path.join(lbl_dir, os.path.splitext(files[i])[0] + ".txt")
        param = {"boxes": load_labels(txt, w, h), "cls": cls_id, "scale": scale}
        cv2.setMouseCallback("Label Tool", on_mouse, param)

        while True:
            view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
            for c, x1, y1, x2, y2 in param["boxes"]:
                col = COLORS[c % len(COLORS)]
                cv2.rectangle(view, (int(x1 * scale), int(y1 * scale)),
                              (int(x2 * scale), int(y2 * scale)), col, 2)
                cv2.putText(view, classes[c] if c < len(classes) else str(c),
                            (int(x1 * scale), int(y1 * scale) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if state["drag"] and state["cur"]:
                cv2.rectangle(view, state["drag"], state["cur"], (255, 255, 255), 1)
            name = classes[param["cls"]] if param["cls"] < len(classes) else str(param["cls"])
            cv2.putText(view, f"[{i + 1}/{len(files)}] class={param['cls']}:{name}  boxes={len(param['boxes'])}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Label Tool", view)

            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k in (ord("q"), 27):
                save_labels(txt, param["boxes"], w, h)
                cv2.destroyAllWindows()
                print("已保存，退出。标注文件在", lbl_dir)
                return
            if k in (ord(" "), ord("n"), 83, 13):
                save_labels(txt, param["boxes"], w, h)
                i += 1
                break
            if k in (ord("p"), 81) and i > 0:
                save_labels(txt, param["boxes"], w, h)
                i -= 1
                break
            if k == ord("d") and param["boxes"]:
                param["boxes"].pop()
            if ord("0") <= k <= ord("9"):
                cls_id = k - ord("0")
                if cls_id < len(classes):
                    param["cls"] = cls_id
                else:
                    cls_id = param["cls"]

    cv2.destroyAllWindows()
    print("全部标完！标注文件在", lbl_dir)
    input("按回车退出")


if __name__ == "__main__":
    main()
