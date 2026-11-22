"""USB 相机打开与格式协商（CLI / GUI 共用）。

水下实验最常见的问题不是算法，而是**相机没跑到预期帧率**：

* UVC 相机通常有两个 ``/dev/videoX``：一个是取流节点，另一个是 metadata 节点，
  打开 metadata 节点会失败或没有图像；
* 1280x720 下 ``YUYV`` 未压缩，USB2 带宽只能到 10 fps（部分相机 5 fps），
  必须切 ``MJPG`` 才能到 30 fps；
* 低光照时自动曝光会拉长曝光时间，并可能触发 ``exposure_dynamic_framerate``，
  导致帧率进一步下降。

因此所有打开 USB 相机的地方都走这里，保证“先设 FOURCC，再设分辨率/FPS”。
"""

from __future__ import annotations

import os
import re
import sys

import cv2

__all__ = ["fourcc_to_str", "open_camera", "camera_info", "DEFAULT_FOURCC", "DEFAULT_FPS"]

DEFAULT_FOURCC = "MJPG"
DEFAULT_FPS = 30.0


def fourcc_to_str(value: int) -> str:
    """把 OpenCV 的 FOURCC 整数转成人类可读的四字符（例如 ``MJPG``）。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return "".join(chr((number >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


def _backend() -> int:
    """Linux 用 V4L2 以便可靠设置 MJPG；其他平台用默认后端。"""
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(
    index: "int | str",
    width: int = 0,
    height: int = 0,
    fourcc: str = DEFAULT_FOURCC,
    fps: float = DEFAULT_FPS,
) -> cv2.VideoCapture:
    """打开 USB/CSI 相机并尽量协商到 ``fourcc + 分辨率 + fps``。

    顺序很重要：**先设 FOURCC，再设宽高，最后设 FPS**。反过来的话，
    驱动可能已经按 YUYV 完成了格式协商，再改 FOURCC 会失败或被忽略。

    Args:
        index: 相机序号（``/dev/video0`` -> 0），或设备路径
            （``/dev/v4l/by-id/...``）。UVC 相机重新枚举后序号会变化，
            长期实验建议用 by-id 路径。注意 metadata 节点打不开。
        width/height: 目标分辨率；0 表示不设置。
        fourcc: ``MJPG``（推荐）或 ``YUYV``。
        fps: 目标帧率；0 表示不设置。

    Returns:
        ``cv2.VideoCapture``；调用方用 ``isOpened()`` 判断是否成功。
    """
    source: "int | str"
    if isinstance(index, str):
        text = index.strip()
        if text.startswith("/"):
            # OpenCV 的 V4L2 后端不认 /dev/v4l/by-id/... 这种别名，
            # 先解析 symlink 到真实的 /dev/videoN，再按索引打开。
            real = os.path.realpath(text)
            match = re.search(r"video(\d+)$", real)
            source = int(match.group(1)) if match else real
        elif text.lstrip("-").isdigit():
            source = int(text)
        else:
            source = text
    else:
        source = int(index)
    capture = cv2.VideoCapture(source, _backend())
    if not capture.isOpened():
        return capture

    code = str(fourcc or "").strip().upper()
    if len(code) == 4:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))

    if int(width) > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if int(height) > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if float(fps) > 0.0:
        capture.set(cv2.CAP_PROP_FPS, float(fps))

    # 只保留最新帧，降低“读到旧帧”的延迟（某些驱动不支持，失败无妨）
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def camera_info(capture: cv2.VideoCapture) -> dict:
    """读取相机实际协商结果，用于日志与故障判断。"""
    if capture is None or not capture.isOpened():
        return {"opened": False}
    return {
        "opened": True,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "fourcc": fourcc_to_str(int(capture.get(cv2.CAP_PROP_FOURCC))),
    }
