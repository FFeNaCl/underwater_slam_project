"""真实环境 YOLO 数据集采集工具（不依赖 ROS）。

用途：把 USB 相机或视频里的**真实水下画面**按固定间隔抽帧保存，
后续用 labelImg / X-AnyLabeling 标注成 YOLO 格式。

用法::

    # USB 相机，自动每 0.5 s 存一张，最多 300 张
    python3 -m underwater_perception.tools.capture_dataset \
      --camera 0 --out ~/underwater_slam_project/datasets/real_underwater/raw \
      --interval 0.5 --max 300 --names cube,marker,pipe

    # 视频文件抽帧（不打开窗口，适合远程/无显示器）
    python3 -m underwater_perception.tools.capture_dataset \
      --video pool.mp4 --no-display --interval 0.5 --max 500 \
      --out ~/datasets/real_underwater/raw

    # 只用键盘手动挑帧（画面窗口）
    python3 -m underwater_perception.tools.capture_dataset --camera 0 --manual

按键（有显示窗口时）：

* ``SPACE`` 立即保存
* ``A`` 自动采集开/关
* ``Q`` / ``ESC`` 退出

输出结构（可直接作为 YOLO 数据集的根目录，标注后把 images/labels 按
train/valid 拆开即可）::

    raw/
    ├── images/            # 原始帧（jpg/png）
    ├── labels/            # --write-empty-labels 时生成同名空 txt，方便标注工具直接打开
    ├── data.yaml          # 给了 --names 时生成模板
    └── capture_session.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2

from ..core.camera_io import DEFAULT_FOURCC, DEFAULT_FPS, camera_info, open_camera
from ..core.dataset_io import (
    ensure_dataset_dirs,
    frame_filename,
    imwrite_unicode,
    write_data_yaml,
    write_session_json,
)

__all__ = ["main"]

WINDOW_NAME = "capture_dataset"


def _parse_names(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _open_source(args: argparse.Namespace):
    if args.video:
        capture = cv2.VideoCapture(str(Path(args.video).expanduser()))
        label = str(args.video)
    else:
        capture = open_camera(
            int(args.camera),
            width=int(args.width),
            height=int(args.height),
            fourcc=str(args.fourcc),
            fps=float(args.fps),
        )
        label = f"camera{int(args.camera)}"
    if not capture.isOpened():
        hint = ""
        if not args.video:
            hint = (
                "；Linux 下 /dev/video1、/dev/video3 常是 UVC metadata 节点，"
                "请改用 video0/video2 对应的序号"
            )
        raise RuntimeError(f"无法打开输入源: {label}{hint}")
    return capture, label


def _headless(args: argparse.Namespace) -> bool:
    return bool(args.no_display) or not os.environ.get("DISPLAY")


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="真实环境 YOLO 数据集采集（USB 相机 / 视频）")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="USB/CSI 相机序号（Linux /dev/video0 -> 0）")
    source.add_argument("--video", default="", help="视频文件路径")
    parser.add_argument("--out", default="~/underwater_slam_project/datasets/real_underwater/raw", help="数据集根目录")
    parser.add_argument("--interval", type=float, default=0.5, help="自动采集间隔（秒）")
    parser.add_argument("--max", type=int, default=300, help="最多保存多少张（0=不限）")
    parser.add_argument("--width", type=int, default=1280, help="相机宽度（0=不设置）")
    parser.add_argument("--height", type=int, default=720, help="相机高度（0=不设置）")
    parser.add_argument("--fourcc", default=DEFAULT_FOURCC, choices=["MJPG", "YUYV"], help="像素格式：MJPG 才有 30 fps")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="相机目标帧率（USB 相机通常 30）")
    parser.add_argument("--prefix", default="frame", help="文件名前缀")
    parser.add_argument("--format", default="jpg", choices=["jpg", "jpeg", "png", "bmp"], help="保存格式")
    parser.add_argument("--names", default="", help="类别名，逗号分隔；给了就生成 data.yaml 模板")
    parser.add_argument("--manual", action="store_true", help="默认自动采集；加此参数改为只用 SPACE 手动保存")
    parser.add_argument("--write-empty-labels", action="store_true", help="同时生成同名空 txt（背景图标注用）")
    parser.add_argument("--no-display", action="store_true", help="不打开预览窗口（远程/无显示器）")
    args = parser.parse_args(argv)

    try:
        images_dir, labels_dir = ensure_dataset_dirs(args.out)
        capture, source_label = _open_source(args)
    except (RuntimeError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 3

    if not args.video:
        info = camera_info(capture)
        print(
            "相机实际格式: {fourcc} {width}x{height} @ {fps:.1f} fps".format(**info)
        )
        if info.get("fourcc") != "MJPG" and int(args.width) >= 1280:
            print(
                "[警告] 当前不是 MJPG，1280x720 的 YUYV 在 USB2 上通常只有 5~10 fps；"
                "请确认 --fourcc MJPG"
            )

    root = Path(args.out).expanduser()
    names = _parse_names(args.names)
    if names:
        print(f"已生成 data.yaml 模板: {write_data_yaml(root, names)}")

    headless = _headless(args)
    auto = not args.manual
    saved = 0
    attempts = 0
    last_save = 0.0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"输入 {source_label} -> {images_dir}")
    print(
        "自动采集 %s，间隔 %.2f s，最多 %s 张%s"
        % ("开启" if auto else "关闭", args.interval, args.max if args.max > 0 else "不限", "（无窗口）" if headless else "")
    )
    if not headless:
        print("按键：SPACE=保存  A=自动开关  Q/ESC=退出")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[提示] 输入源已结束或读帧失败")
                break
            attempts += 1
            now = time.time()

            should_save = False
            if auto and (now - last_save) >= float(args.interval):
                should_save = True

            if not headless:
                preview = frame.copy()
                text = (
                    f"saved {saved} | attempts {attempts} | auto {'ON' if auto else 'OFF'} | "
                    f"interval {args.interval:.2f}s"
                )
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 60), (18, 32, 42), -1)
                cv2.putText(preview, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 245, 250), 1, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    "SPACE save | A auto | Q quit",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (80, 210, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    should_save = True
                if key == ord("a"):
                    auto = not auto
                    print(f"自动采集 {'开启' if auto else '关闭'}")
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if not should_save:
                continue

            stamp_ns = int(time.time() * 1e9)
            filename = frame_filename(args.prefix, saved, stamp_ns, args.format)
            target = images_dir / filename
            if not imwrite_unicode(target, frame):
                print(f"[警告] 写入失败: {target}", file=sys.stderr)
                continue
            if args.write_empty_labels:
                (labels_dir / f"{Path(filename).stem}.txt").write_text("", encoding="utf-8")
            saved += 1
            last_save = now
            if saved % 10 == 0:
                print(f"  已保存 {saved} 张（最近: {filename}）")
            if args.max > 0 and saved >= args.max:
                print(f"已达到上限 {args.max} 张，退出。")
                break
    finally:
        capture.release()
        if not headless:
            cv2.destroyWindow(WINDOW_NAME)

    session = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source_label,
        "image_dir": str(images_dir),
        "label_dir": str(labels_dir),
        "saved": int(saved),
        "attempts": int(attempts),
        "interval_s": float(args.interval),
        "image_format": args.format,
        "classes": names,
        "note": "训练/验证集要按视频或场次拆分，不要用相邻帧随机拆分（会泄漏）。",
    }
    session_path = write_session_json(root, session)
    print("")
    print(f"采集结束：保存 {saved} 张 -> {images_dir}")
    print(f"会话记录: {session_path}")
    print("下一步：用 labelImg / X-AnyLabeling 标注为 YOLO 格式，再运行 yolo_dataset_check 检查数据集。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
