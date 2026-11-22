"""OpenCV 4.x / 5.x 兼容层。

本项目的开发机（Windows）装的是 OpenCV 5.0.0，而目标机 Ubuntu 22.04 + ROS2 Humble
是 OpenCV 4.5.4。这两个大版本之间以下 API 有差异，必须统一在这里处理：

* ``cv2.findContours``：OpenCV 3.x 返回 3 个值，4.x/5.x 返回 2 个值 —— 用返回值个数判断。
* ``cv2.findChessboardCornersSB``：4.5+ 才有，精度优于经典版本；缺失时回退。
* ``cv2.aruco``（ChArUco）：4.7 起改用 ``CharucoDetector`` 类，4.6 及以前用
  ``interpolateCornersCharuco``；两者都可能不可用（未编译 contrib），因此 ChArUco
  是"尽力支持"，标定主路径始终是棋盘格。
* ``cv2.solvePnP`` 的 ``SOLVEPNP_IPPE_SQUARE``：4.x/5.x 都有，但为了兼容旧版本仍做回退。

凡是调用上面任何一个，都请通过本模块，不要在别处直接调用，否则会出现
"开发机能跑、目标机报 AttributeError" 的问题。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "find_contours",
    "find_chessboard_corners",
    "has_charuco",
    "solve_pnp_square",
    "opencv_version",
    "opencv_major",
]


def opencv_version() -> str:
    """返回 OpenCV 版本字符串，例如 ``4.5.4`` / ``5.0.0``。"""
    return str(cv2.__version__)


def opencv_major() -> int:
    """返回 OpenCV 主版本号（4 或 5）。"""
    try:
        return int(str(cv2.__version__).split(".")[0])
    except (ValueError, IndexError):  # pragma: no cover - 防御性
        return 4


def find_contours(
    binary: np.ndarray,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """兼容 OpenCV 3/4/5 的 ``findContours`` 包装。

    Args:
        binary: 单通道二值图（0/255）。
        mode: 轮廓检索模式，默认 ``RETR_EXTERNAL``。
        method: 轮廓逼近方法，默认 ``CHAIN_APPROX_SIMPLE``。

    Returns:
        ``(contours, hierarchy)``；hierarchy 在旧版本不存在时为 ``None``。
    """
    result = cv2.findContours(binary, mode, method)
    if len(result) == 2:
        contours, hierarchy = result
        return list(contours), hierarchy
    # OpenCV 3.x: (image, contours, hierarchy)
    _, contours, hierarchy = result
    return list(contours), hierarchy


def find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: Tuple[int, int],
    use_sb: bool = True,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """检测棋盘格内角点，优先使用精度更高的 ``findChessboardCornersSB``。

    Args:
        gray: 单通道灰度图。
        pattern_size: ``(内角点列数, 内角点行数)``，例如 8x6 棋盘格传 ``(8, 6)``。
        use_sb: 是否优先尝试 SB 算法（对模糊、光照不均更稳）。

    Returns:
        ``(found, corners, method)``；``corners`` 形状为 ``(N, 1, 2)`` float32，
        ``method`` 为实际使用的算法名（``"SB"`` / ``"CLASSIC"`` / ``"NONE"``）。
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
            if found and corners is not None and len(corners) > 0:
                return True, np.asarray(corners, dtype=np.float32), "SB"
        except cv2.error:  # pragma: no cover - 某些构建缺少 SB
            pass

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if found and corners is not None and len(corners) > 0:
        return True, np.asarray(corners, dtype=np.float32), "CLASSIC"
    return False, None, "NONE"


def has_charuco() -> bool:
    """ChArUco 是否可用（需要 opencv-contrib，且 API 版本不一）。"""
    return hasattr(cv2, "aruco")


def solve_pnp_square(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
    """已知正方形（立方体可见面）四角点求位姿，优先 IPPE_SQUARE。

    **必须校验解**（踩过的坑）：``SOLVEPNP_IPPE_SQUARE`` 对物体点的顺序有严格要求
    （``(-s/2,+s/2,0)``、``(+s/2,+s/2,0)``、``(+s/2,-s/2,0)``、``(-s/2,-s/2,0)``）。
    顺序不匹配时它 **返回 ``ok=True`` 但给出退化解**（实测平移为 ``[0,0,-0]``），
    调用方直接采信就会得到完全错误的结果。因此这里对每个候选方法都校验：
    平移非退化、深度为正、重投影误差可接受；不满足就换下一个方法
    （``ITERATIVE`` 对顺序不敏感，但不校验也会返回噪声解）。

    Args:
        object_points: ``(4, 3)`` 物体坐标系角点（正方形，边长已知，z=0）。
        image_points: ``(4, 1, 2)`` 或 ``(4, 2)`` 图像坐标。
        camera_matrix: ``(3, 3)`` 内参。
        dist_coeffs: 畸变系数或 ``None``。

    Returns:
        ``(ok, rvec, tvec, method)``；所有方法都不通过校验时 ``ok=False``。
    """
    obj = np.asarray(object_points, dtype=np.float64).reshape(4, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(4, 1, 2)
    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    methods: List[Tuple[str, int]] = []
    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        methods.append(("IPPE_SQUARE", cv2.SOLVEPNP_IPPE_SQUARE))
    methods.append(("ITERATIVE", cv2.SOLVEPNP_ITERATIVE))

    for name, flag in methods:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, camera_matrix, dist, flags=flag, useExtrinsicGuess=False
            )
        except cv2.error:
            continue
        if not ok or rvec is None or tvec is None:
            continue
        tvec_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(tvec_flat)) or tvec_flat[2] <= 1e-6:
            continue  # 退化解（IPPE 顺序不匹配的典型症状）或面在相机后方
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist)
        residual = projected.reshape(4, 2) - img.reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if error > 5.0:
            continue
        return True, rvec, tvec, name
    return False, None, None, "NONE"


def as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """把任意掩码规范为 ``uint8`` 0/255 单通道图。"""
    if mask.dtype == np.uint8 and mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 或灰度输入统一转单通道灰度。"""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
