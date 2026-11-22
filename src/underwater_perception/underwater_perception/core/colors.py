"""颜色分割：几何旁路的第一级证据。

水下成像的两个事实决定了这里的默认值：

1. **红光衰减最快**，红色物体在 1~2 m 外会变成暗橙/褐色，饱和度也会掉。因此
   ``red`` 预设的 S/V 下限取得比较低（90/55），并且把 0 度附近的区间放宽到 H≤10。
2. **水体偏蓝绿**，白色/银色反光容易带上蓝绿，但它们不符合"R 通道显著大于 G/B"
   这一条件，所以 :func:`red_dominance_mask` 能有效压掉这类误检。

调试手法：``cube_detect_cli --print-mask out.png`` 会把 mask 和叠加图存下来，
比在代码里猜阈值快得多。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .params import ColorConfig
from .types import GeometryMetrics

__all__ = [
    "build_color_mask",
    "red_dominance_mask",
    "mask_statistics",
    "clean_mask",
    "hsv_of_point",
]


def _odd(value: int) -> int:
    """把核大小规范为奇数（<=1 表示关闭）。"""
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def build_color_mask(image: np.ndarray, config: ColorConfig) -> np.ndarray:
    """按 HSV 区间构造目标颜色掩码。

    Args:
        image: BGR 图像（``uint8``，HWC）。
        config: 颜色参数；``hsv_ranges`` 优先于 ``preset``。

    Returns:
        ``uint8`` 掩码，目标区域为 255。
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    working = image
    blur = _odd(config.blur_kernel)
    if blur:
        working = cv2.GaussianBlur(image, (blur, blur), 0)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for h_low, s_low, v_low, h_high, s_high, v_high in config.resolved_ranges():
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    if config.min_saturation > 0 or config.min_value > 0:
        extra_lower = np.array([0, int(config.min_saturation), int(config.min_value)], dtype=np.uint8)
        extra_upper = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_and(mask, cv2.inRange(hsv, extra_lower, extra_upper))

    if config.require_red_dominance:
        mask = cv2.bitwise_and(mask, red_dominance_mask(image, config.red_dominance_ratio))

    return clean_mask(mask, config)


def red_dominance_mask(image: np.ndarray, ratio: float = 1.15) -> np.ndarray:
    """返回"红色通道占优"的掩码：``R > ratio*G 且 R > ratio*B``。

    这个条件与具体色相无关，因此能同时容纳水下偏橙、偏褐的红块，
    又能排除水面白亮反光（三通道接近）和蓝绿背景。

    Args:
        image: BGR 图像。
        ratio: 比值门限，>1 才有效；<=1 时退化为 ``R >= G`` 且 ``R >= B``。

    Returns:
        ``uint8`` 掩码（255 表示通过）。
    """
    if image.ndim == 2:
        return np.full(image.shape, 255, dtype=np.uint8)

    b = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)
    if ratio <= 1.0:
        ok = (r >= g) & (r >= b)
    else:
        ok = (r.astype(np.float32) > ratio * g.astype(np.float32)) & (
            r.astype(np.float32) > ratio * b.astype(np.float32)
        )
    return np.where(ok, 255, 0).astype(np.uint8)


def clean_mask(mask: np.ndarray, config: ColorConfig) -> np.ndarray:
    """形态学清理：先开运算去噪点，再闭运算填内部空洞。"""
    out = mask
    open_k = _odd(config.morph_open_kernel)
    if open_k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    close_k = _odd(config.morph_close_kernel)
    if close_k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def mask_statistics(hsv: np.ndarray, mask: np.ndarray, polygon: Sequence[Sequence[float]] | None = None) -> Tuple[float, float, float, float]:
    """统计掩码（可选限定在多边形内）的色相/饱和度/亮度均值与颜色纯度。

    Args:
        hsv: HSV 图像。
        mask: 颜色掩码。
        polygon: 可选多边形（轮廓），只统计其内部像素。

    Returns:
        ``(color_purity, mean_hue, mean_saturation, mean_value)``；
        ``color_purity`` 为多边形内属于 mask 的像素占比（多边形缺失时按整帧统计）。
    """
    region = mask
    if polygon is not None and len(polygon) >= 3:
        inside = np.zeros(mask.shape[:2], dtype=np.uint8)
        cv2.fillPoly(inside, [np.asarray(polygon, dtype=np.int32)], 255)
        region = cv2.bitwise_and(mask, inside)
        total = int(np.count_nonzero(inside))
        purity = float(np.count_nonzero(region) / total) if total > 0 else 0.0
        sel = region > 0
    else:
        sel = mask > 0
        purity = 1.0 if np.any(sel) else 0.0

    if not np.any(sel):
        return purity, 0.0, 0.0, 0.0

    pixels = hsv[sel]
    return (
        purity,
        float(np.mean(pixels[:, 0])),
        float(np.mean(pixels[:, 1])),
        float(np.mean(pixels[:, 2])),
    )


def hsv_of_point(hsv: np.ndarray, point: Sequence[float]) -> Tuple[int, int, int]:
    """取某点的 HSV 值（越界时返回 ``(0, 0, 0)``），用于现场取色调试。"""
    u = int(round(float(point[0])))
    v = int(round(float(point[1])))
    if not (0 <= v < hsv.shape[0] and 0 <= u < hsv.shape[1]):
        return (0, 0, 0)
    h, s, val = hsv[v, u]
    return int(h), int(s), int(val)
