"""ROS 图像消息 <-> numpy 转换（不依赖 cv_bridge，避免版本/编码坑）。

为什么不用 ``cv_bridge``：``cv_bridge`` 的可用编码与 OpenCV 版本绑定，
在 Humble 上遇到 ``16UC1``/``rgb8`` 等情况时常抛异常；而本项目只需要
bgr8/rgb8/bgra8/mono8/mono16 这几种，自己转更可控、也更容易在无 ROS 环境里单测。
如果需要 cv_bridge，可以在节点里改用 ``cv_bridge.CvBridge``，本模块的接口保持不变。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sensor_msgs.msg import Image

__all__ = ["msg_to_bgr", "bgr_to_msg", "stamp_to_seconds", "SUPPORTED_ENCODINGS"]

#: 编码 -> (通道数, 每通道字节数)
SUPPORTED_ENCODINGS: Dict[str, tuple] = {
    "bgr8": (3, 1),
    "rgb8": (3, 1),
    "bgra8": (4, 1),
    "rgba8": (4, 1),
    "mono8": (1, 1),
    "8UC1": (1, 1),
    "8UC3": (3, 1),
    "8UC4": (4, 1),
    "mono16": (1, 2),
    "16UC1": (1, 2),
    "bayer_rggb8": (1, 1),
    "bayer_bggr8": (1, 1),
    "bayer_gbrg8": (1, 1),
    "bayer_grbg8": (1, 1),
}


def msg_to_bgr(msg: Image) -> np.ndarray:
    """``sensor_msgs/Image`` -> BGR ``uint8`` 图像（HWC）。

    灰度/Bayer/16 位输入统一转成 3 通道 BGR，保证下游（检测、标定）只有一种输入格式。
    Bayer 只做最简单的去马赛克（``cvtColor`` 的 Bayer 转换），不做 ISP。

    Args:
        msg: ROS 图像消息。

    Returns:
        ``(H, W, 3)`` ``uint8`` BGR 数组。

    Raises:
        ValueError: 编码不在支持列表中，或数据长度与尺寸不匹配。
    """
    encoding = (msg.encoding or "").lower()
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    channels, bytes_per_channel = SUPPORTED_ENCODINGS[encoding]

    expected = int(msg.height) * int(msg.width) * channels * bytes_per_channel
    if len(msg.data) < expected and int(msg.step) * int(msg.height) > len(msg.data):
        raise ValueError(
            f"image data too short: got {len(msg.data)} bytes, need {expected} for {msg.width}x{msg.height} {msg.encoding}"
        )

    dtype = np.uint8 if bytes_per_channel == 1 else np.uint16
    buffer = np.frombuffer(msg.data, dtype=dtype)

    # 逐行按 step 取，兼容带 padding 的发布者（step > width*channels*bytes）
    row_values = int(msg.step) // bytes_per_channel if int(msg.step) > 0 else int(msg.width) * channels
    if row_values == int(msg.width) * channels:
        image = buffer[: int(msg.height) * row_values].reshape((int(msg.height), int(msg.width), channels))
    else:
        rows = []
        for index in range(int(msg.height)):
            start = index * row_values
            rows.append(buffer[start : start + int(msg.width) * channels])
        image = np.stack(rows, axis=0).reshape((int(msg.height), int(msg.width), channels))

    image = np.ascontiguousarray(image)

    if bytes_per_channel == 2:
        # 16 位按当前最大值归一化到 8 位（水下相机常见 16UC1 输出）
        peak = float(image.max()) if image.size else 0.0
        scale = 255.0 / peak if peak > 255.0 else 1.0
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    import cv2

    if channels == 1:
        if encoding.startswith("bayer"):
            code = {
                "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
                "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
                "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
                "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
            }[encoding]
            return np.ascontiguousarray(cv2.cvtColor(image, code))
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if encoding in ("rgb8",):
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if encoding in ("rgba8",):
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))
    if encoding in ("bgra8",):
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    return image


def bgr_to_msg(
    image: np.ndarray,
    stamp=None,
    frame_id: str = "",
    encoding: str = "bgr8",
) -> Image:
    """BGR/灰度图像 -> ``sensor_msgs/Image``。

    Args:
        image: ``(H, W, 3)`` BGR 或 ``(H, W)`` 灰度图像（``uint8``）。
        stamp: ``builtin_interfaces/msg/Time``；``None`` 时留空。
        frame_id: 参考坐标系名。
        encoding: 输出编码：``bgr8`` 或 ``mono8``。``mono8`` 时自动转灰度，
            数据量只有 BGR 的 1/3，可显著降低 1280x720 下的 CPU/DDS 开销。

    Returns:
        ``sensor_msgs/msg/Image``。
    """
    import cv2

    output_encoding = (encoding or "bgr8").lower()
    if output_encoding == "mono8":
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.ascontiguousarray(image)
        channels = 1
    else:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.ascontiguousarray(image)
        channels = image.shape[2]

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "mono8" if output_encoding == "mono8" else "bgr8"
    msg.is_bigendian = False
    msg.step = int(image.shape[1] * channels)
    msg.data = image.tobytes()
    return msg


def stamp_to_seconds(header) -> Optional[float]:
    """把消息头时间戳转成秒；为空（0）时返回 ``None``，由调用方决定回退策略。"""
    if header is None:
        return None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    seconds = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
    return seconds if seconds > 0.0 else None
