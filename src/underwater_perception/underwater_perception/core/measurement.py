"""单目测量与多视图三角化。

坐标约定（全文统一，务必先读）：

* **相机光学坐标系**（OpenCV）：``x`` 向右、``y`` 向下、``z`` 向前（光轴）。
  因此像平面点 ``(u, v)`` 对应的射线是 ``K⁻¹ [u, v, 1]ᵀ`` 归一化。
* **方位角**：``azimuth = atan2(x, z)``（向右为正），
  ``elevation = atan2(y, sqrt(x²+z²))``（向下为正）。
* **世界/地图坐标系**：单目 SLAM 的初始化尺度由第一对关键帧的基线决定，
  因此地图单位不是米，而是"第一段基线 = 1"的任意尺度。**只有**在
  ``method == "known_size"``（已知立方体边长）或引入了外部尺度来源时，
  距离才有物理意义。

三个可用的测距手段，可靠性从高到低：

1. ``pose``：SLAM 位姿 + 多视图射线三角化（:func:`triangulate_rays`），
   由语义地标节点使用，是**唯一**同时给出方向和位置且不依赖先验尺寸的方法
   （前提是 SLAM 地图已有尺度）。
2. ``known_size``：已知立方体边长 ``L``，正视假设下 ``Z = fx·L / side_px``。
   单帧即可，误差主要来自"面不正对相机"和边缘模糊，工程上够用。
3. ``bearing_only``：只有方向，没有距离（未标定/未给边长时）。

所有函数都不 import rclpy，可在 Windows 上直接单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .cv_compat import solve_pnp_square
from .types import Detection, GeometryMetrics, Measurement

__all__ = [
    "MeasurementResult",
    "pixel_to_ray",
    "bearing_from_ray",
    "ray_to_world",
    "distance_from_known_size",
    "face_pose_from_quad",
    "measure_geometry",
    "refine_cube_pose",
    "triangulate_rays",
    "pose_to_matrix",
    "matrix_to_pose",
    "invert_pose",
]


@dataclass
class MeasurementResult:
    """测量结果（内部表示，最终写入 :class:`~..types.Measurement`）。"""

    distance_m: Optional[float] = None
    bearing_rad: Optional[Tuple[float, float]] = None
    ray_camera: Optional[np.ndarray] = None
    method: str = "none"
    quality: float = 0.0
    notes: str = ""
    pose_rvec: Optional[np.ndarray] = None
    pose_tvec: Optional[np.ndarray] = None
    fit_error_px: Optional[float] = None

    def to_measurement(self) -> Measurement:
        """转换成消息契约里的 :class:`Measurement`。"""
        return Measurement(
            distance_m=self.distance_m,
            bearing_rad=self.bearing_rad,
            ray_camera=None if self.ray_camera is None else [float(v) for v in self.ray_camera],
            method=self.method,
            quality=float(self.quality),
            pose_rvec=None if self.pose_rvec is None else [float(v) for v in np.asarray(self.pose_rvec).reshape(3)],
            pose_tvec=None if self.pose_tvec is None else [float(v) for v in np.asarray(self.pose_tvec).reshape(3)],
            fit_error_px=None if self.fit_error_px is None else float(self.fit_error_px),
        )


def pixel_to_ray(u: float, v: float, camera_matrix: np.ndarray) -> np.ndarray:
    """像素点 -> 相机坐标系单位射线。

    Args:
        u: 横坐标（像素）。
        v: 纵坐标（像素）。
        camera_matrix: ``(3, 3)`` 内参矩阵。

    Returns:
        长度 3 的单位向量（相机光学坐标系）。
    """
    k_inv = np.linalg.inv(np.asarray(camera_matrix, dtype=np.float64))
    ray = k_inv @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(ray))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return ray / norm


def bearing_from_ray(ray: Sequence[float]) -> Tuple[float, float]:
    """由相机坐标系射线求 ``(方位角, 俯仰角)``，单位弧度。"""
    x, y, z = (float(ray[0]), float(ray[1]), float(ray[2]))
    azimuth = math.atan2(x, z)
    elevation = math.atan2(y, math.hypot(x, z))
    return azimuth, elevation


def ray_to_world(ray_camera: Sequence[float], pose_wc: np.ndarray) -> np.ndarray:
    """把相机坐标系射线旋转到世界坐标系。

    Args:
        ray_camera: 相机坐标系下的方向向量（不必归一化）。
        pose_wc: ``(4, 4)`` 世界<-相机 变换矩阵（即 SLAM 输出的相机位姿）。

    Returns:
        世界坐标系下的单位方向向量。
    """
    rot = np.asarray(pose_wc, dtype=np.float64)[:3, :3]
    direction = rot @ np.asarray(ray_camera, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-12 else direction


def distance_from_known_size(side_px: float, focal_px: float, edge_m: float) -> Optional[float]:
    """正视假设下的已知尺寸测距：``Z = f · L / s``。

    Args:
        side_px: 目标像素边长（由最小外接矩形短边给出）。
        focal_px: 焦距（像素），通常取 ``(fx + fy) / 2``。
        edge_m: 目标真实边长（米）。

    Returns:
        距离（米）；输入非法时返回 ``None``。
    """
    if side_px <= 1e-6 or focal_px <= 1e-6 or edge_m <= 0.0:
        return None
    return float(focal_px * edge_m / side_px)


def face_pose_from_quad(
    corners: Sequence[Sequence[float]],
    edge_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str, float]:
    """用可见面的 4 个角点求立方体面位姿（solvePnP + IPPE_SQUARE）。

    物体坐标系原点取面中心，``x`` 沿一条边、``y`` 沿另一条边、``z`` 指向相机。
    ``tvec`` 即"相机坐标系下的面中心"，其模长是到面中心的距离。

    **为什么这里要搜索角点顺序**（重要，踩过坑）：
    轮廓逼近给出的 4 个角点顺序是任意的（起点任意、绕向任意），而
    ``SOLVEPNP_IPPE_SQUARE`` 要求严格的角点对应顺序。顺序不匹配时 OpenCV
    **不会报错**，而是返回一个看似合理但完全错误的位姿（实测得到 0.005 m，
    而真值是 0.45 m）。因此这里枚举"两种绕向 × 4 个起点"共 8 种合法对应，
    用重投影误差选出正确的那一个，并顺带得到该位姿自身的质量指标。

    Args:
        corners: 4 个角点（任意顺序），像素坐标。
        edge_m: 边长（米）。
        camera_matrix: ``(3, 3)`` 内参。
        dist_coeffs: 畸变系数或 ``None``。

    Returns:
        ``(ok, rvec, tvec, method, reproj_error_px)``；失败时后四项为
        ``(None, None, "NONE", inf)``。
    """
    if len(corners) != 4 or edge_m <= 0.0:
        return False, None, None, "NONE", float("inf")

    half = edge_m / 2.0
    # 与 SOLVEPNP_IPPE_SQUARE 要求一致：左上、右上、右下、左下（y 向下坐标系）
    object_points = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float64,
    )
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]

    best: Optional[Tuple[float, np.ndarray, np.ndarray, str]] = None
    for sequence in (ordered, ordered[::-1]):
        for shift in range(4):
            candidate = np.roll(sequence, -shift, axis=0)
            ok, rvec, tvec, method = solve_pnp_square(object_points, candidate, camera_matrix, dist_coeffs)
            if not ok or rvec is None or tvec is None:
                continue
            tvec_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
            if tvec_flat[2] <= 1e-6:
                continue  # 面在相机后方：非法解
            dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
            residual = projected.reshape(4, 2) - candidate
            error = float(math.sqrt(float(np.mean(np.sum(residual * residual, axis=1)))))
            if best is None or error < best[0]:
                best = (error, np.asarray(rvec), tvec_flat, method)

    if best is None:
        return False, None, None, "NONE", float("inf")
    error, rvec, tvec, method = best
    return True, rvec, tvec, method, error


def measure_geometry(
    geometry: GeometryMetrics,
    confidence: float,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray] = None,
    edge_m: float = 0.05,
    polygon: Optional[Sequence[Sequence[float]]] = None,
    min_side_px: float = 12.0,
    use_silhouette_fit: bool = True,
    fit_max_rms_px: float = 5.0,
) -> MeasurementResult:
    """对单个几何观测做测量（射线 + 可用时的距离与 6DoF 位姿）。

    距离估计的优先级（前面的失败或超门限才退到后面）：

    1. ``known_size_pose``：4 顶点正视轮廓 + ``solvePnP(IPPE_SQUARE)``，重投影 ≤2 px；
    2. ``known_size_refined``：轮廓拟合（ICP + PnP，多起点），残差 ≤``fit_max_rms_px``；
    3. ``known_size``：正视假设 ``Z = f·L/s``（斜视时会有 ~25% 系统误差）；
    4. ``bearing_only``：只知道方向。

    Args:
        geometry: 检测到的几何指标。
        confidence: 检测置信度（用于质量评分）。
        camera_matrix: 内参；``None`` 时只能返回像素级信息。
        dist_coeffs: 畸变系数。
        edge_m: 立方体真实边长（米），<=0 表示不知道尺寸。
        polygon: 轮廓点；``None`` 时用 ``geometry.corners``。
        min_side_px: 小于该像素边长时不给距离。
        use_silhouette_fit: 是否启用轮廓拟合（精度换速度的开关）。
        fit_max_rms_px: 轮廓拟合残差门限（像素）。

    Returns:
        :class:`MeasurementResult`。
    """
    result = MeasurementResult()
    if camera_matrix is None:
        result.notes = "camera_matrix_missing"
        return result

    u, v = geometry.center
    ray = pixel_to_ray(u, v, camera_matrix)
    result.ray_camera = ray
    result.bearing_rad = bearing_from_ray(ray)

    focal = 0.5 * (float(camera_matrix[0, 0]) + float(camera_matrix[1, 1]))
    size_score = max(0.0, min(1.0, geometry.side_px / max(min_side_px * 4.0, 1.0)))
    result.quality = float(max(0.0, min(1.0, 0.5 * confidence + 0.5 * size_score)))

    if edge_m <= 0.0 or geometry.side_px < min_side_px:
        result.method = "bearing_only"
        result.notes = "size_unknown_or_too_small"
        return result

    # 优先用 4 角点求位姿（对面正视时更准），否则退化为已知尺寸测距
    corners = list(polygon) if polygon is not None else list(geometry.corners)
    if len(corners) == 4:
        ok, rvec, tvec, method, pose_error = face_pose_from_quad(corners, edge_m, camera_matrix, dist_coeffs)
        # 重投影误差门限：>2 px 说明角点对应关系或轮廓本身不可靠（例如六边形被逼近成 4 点）
        if ok and tvec is not None and pose_error <= 2.0:
            # 【关键】face_pose_from_quad 的物体原点是**可见面中心**，不是立方体中心。
            # 直接取 |tvec| 会系统性偏小约半个边长（5 cm 立方体 = 2.5 cm，约 5.6%）。
            # 但**不能**固定加 +h：平面靶标的 PnP 存在法向二义性（把面翻过来投影误差一样），
            # 所以 ±h 两个候选都可能出现。用物理事实选解——立方体在可见面之后，
            # 因此其中心相对相机必然更远（z 更大）。
            half_edge = 0.5 * float(edge_m)
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            tvec_face = np.asarray(tvec, dtype=np.float64).reshape(3)
            offset = rotation @ np.array([0.0, 0.0, half_edge], dtype=np.float64)
            candidate_a = tvec_face + offset
            candidate_b = tvec_face - offset
            tvec_center = candidate_a if candidate_a[2] >= candidate_b[2] else candidate_b
            distance = float(np.linalg.norm(tvec_center))
            if distance > 0.0:
                result.distance_m = distance
                result.method = "known_size_pose"
                result.notes = f"solvePnP:{method},reproj={pose_error:.2f}px,center_corrected"
                result.pose_rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
                result.pose_tvec = tvec_center
                result.fit_error_px = float(pose_error)
                # 位姿解自身的重投影误差越小，质量越高
                result.quality = float(max(0.0, min(1.0, result.quality * max(0.2, 1.0 - pose_error / 2.0))))
                return result

    # 斜视（5~6 顶点轮廓）时"正视假设"会有 ~25~30% 误差，改用轮廓拟合求 6DoF 位姿。
    if use_silhouette_fit and len(corners) >= 4:
        z_seed = distance_from_known_size(geometry.side_px, focal, edge_m)
        ok_fit, rvec_fit, tvec_fit, fit_error = refine_cube_pose(
            corners, edge_m, camera_matrix, dist_coeffs, z_init=z_seed
        )
        if ok_fit and tvec_fit is not None and fit_error <= float(fit_max_rms_px):
            distance = float(np.linalg.norm(tvec_fit.reshape(3)))
            if 0.0 < distance < 1e4:
                result.distance_m = distance
                result.method = "known_size_refined"
                result.notes = f"silhouette_fit,rms={fit_error:.2f}px,seed={z_seed if z_seed else 'none'}"
                result.pose_rvec = np.asarray(rvec_fit, dtype=np.float64).reshape(3)
                result.pose_tvec = np.asarray(tvec_fit, dtype=np.float64).reshape(3)
                result.fit_error_px = float(fit_error)
                result.quality = float(
                    max(0.0, min(1.0, result.quality * max(0.2, 1.0 - fit_error / max(fit_max_rms_px, 1e-6))))
                )
                return result

    distance = distance_from_known_size(geometry.side_px, focal, edge_m)
    if distance is not None:
        result.distance_m = distance
        result.method = "known_size"
        result.notes = "fronto_parallel_assumption"
    else:
        result.method = "bearing_only"
    return result


def triangulate_rays(
    origins: Sequence[Sequence[float]],
    directions: Sequence[Sequence[float]],
) -> Tuple[Optional[np.ndarray], float, int]:
    """多视图射线三角化（线性最小二乘，返回点、RMS 角度误差、使用视图数）。

    求解 ``min_P Σ ||(I − d_i d_iᵀ)(P − o_i)||²``，等价于解

        (Σ (I − d_i d_iᵀ)) P = Σ (I − d_i d_iᵀ) o_i

    该式在射线方向不平行时良态；本函数额外返回 RMS 三角化误差（米/地图单位），
    供调用方按 ``max_reproj_error_px`` 之类的门限过滤。

    Args:
        origins: 每条射线的起点（相机光心在世界坐标系）。
        directions: 每条射线的方向（会自动归一化）。

    Returns:
        ``(point, rms_error, n_views)``；视图不足或退化时 ``point`` 为 ``None``。
    """
    if len(origins) != len(directions) or len(origins) < 2:
        return None, float("inf"), len(origins)

    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    dirs: List[np.ndarray] = []
    for origin, direction in zip(origins, directions):
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            return None, float("inf"), 0
        d = d / norm
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        projector = np.eye(3) - np.outer(d, d)
        a += projector
        b += projector @ o
        dirs.append(d)

    # 射线近平行时 A 接近奇异：用条件数判断，避免给出荒谬解
    if np.linalg.matrix_rank(a, tol=1e-9) < 3:
        return None, float("inf"), len(dirs)
    try:
        point = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:  # pragma: no cover - 上面的 rank 检查已覆盖
        return None, float("inf"), len(dirs)

    errors = []
    for origin, d in zip(origins, dirs):
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        delta = point - o
        along = float(np.dot(delta, d))
        perp = float(np.linalg.norm(delta - along * d))
        errors.append(perp)
    rms = float(math.sqrt(sum(e * e for e in errors) / len(errors)))
    return point, rms, len(dirs)


def pose_to_matrix(position: Sequence[float], orientation_xyzw: Sequence[float]) -> np.ndarray:
    """由位置 + 四元数 ``(x, y, z, w)`` 组装 ``(4, 4)`` 变换矩阵。"""
    x, y, z, w = (float(v) for v in orientation_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        rot = np.eye(3)
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [float(position[0]), float(position[1]), float(position[2])]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> Tuple[List[float], List[float]]:
    """``(4, 4)`` 变换矩阵 -> ``(位置, 四元数 xyzw)``。"""
    m = np.asarray(matrix, dtype=np.float64)
    position = [float(m[0, 3]), float(m[1, 3]), float(m[2, 3])]
    rot = m[:3, :3]
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return position, [float(x), float(y), float(z), float(w)]


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    """刚体变换求逆（比 ``np.linalg.inv`` 更稳、更快）。"""
    m = np.asarray(matrix, dtype=np.float64)
    rot = m[:3, :3]
    trans = m[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def detection_ray_camera(detection: Detection, camera_matrix: np.ndarray) -> np.ndarray:
    """从检测结果取相机坐标系射线（优先用消息里已有的 ``measurement.ray_camera``）。"""
    if detection.measurement is not None and detection.measurement.ray_camera is not None:
        ray = np.asarray(detection.measurement.ray_camera, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(ray))
        if norm > 1e-12:
            return ray / norm
    u, v = detection.center
    return pixel_to_ray(u, v, camera_matrix)


# --------------------------------------------------------------------------- 轮廓拟合
#: 单位立方体（边长 1）的 8 个顶点，与 core.synthetic.cube_object_points 的顺序一致
_UNIT_CUBE = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)


def _project_cube(rvec: np.ndarray, tvec: np.ndarray, edge_m: float, camera_matrix: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """投影立方体 8 顶点，返回 ``(8, 2)`` 像素坐标。"""
    points = _UNIT_CUBE * float(edge_m)
    projected, _ = cv2.projectPoints(points, rvec.reshape(3, 1), tvec.reshape(3, 1), camera_matrix, dist)
    return projected.reshape(8, 2)


def _point_to_polygon_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """每一行点到多边形边界的最短距离（像素），返回 ``(N,)``。

    多边形按闭合折线处理（点到线段距离取最小），因此对凹多边形也成立。
    """
    count = len(polygon)
    if count < 2:
        return np.full(len(points), np.inf)
    best = np.full(len(points), np.inf, dtype=np.float64)
    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        segment = b - a
        length_sq = float(segment @ segment)
        if length_sq < 1e-12:
            distance = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip(((points - a) @ segment) / length_sq, 0.0, 1.0)
            closest = a + t[:, None] * segment
            distance = np.linalg.norm(points - closest, axis=1)
        best = np.minimum(best, distance)
    return best


#: 轮廓拟合时每条轮廓重采样的点数（固定长度残差向量的关键）
SILHOUETTE_SAMPLES = 12


def _resample_polygon(polygon: np.ndarray, count: int = SILHOUETTE_SAMPLES) -> np.ndarray:
    """把闭合多边形按弧长均匀重采样成 ``count`` 个点。

    为什么必须重采样（踩过的坑）：立方体投影轮廓的**顶点数会随位姿改变**
    （4/5/6 个顶点），如果残差向量直接由顶点构成，长度就会变化，
    导致代价函数在迭代之间不可比较，LM 会拒绝所有步长、一步都不动。
    按弧长重采样后残差长度恒定，代价连续可比。
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((count, 2), dtype=np.float64)
    if len(points) < 3:
        return np.repeat(points[:1], count, axis=0)

    closed = np.vstack([points, points[:1]])
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.repeat(points[:1], count, axis=0)

    targets = np.linspace(0.0, total, count, endpoint=False)
    out = np.zeros((count, 2), dtype=np.float64)
    for index, target in enumerate(targets):
        segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(segments) - 1)
        length = max(float(segments[segment_index]), 1e-9)
        fraction = (float(target) - float(cumulative[segment_index])) / length
        out[index] = closed[segment_index] + fraction * (closed[segment_index + 1] - closed[segment_index])
    return out


def _silhouette_residuals(
    observed: np.ndarray,
    projected: np.ndarray,
    samples: int = SILHOUETTE_SAMPLES,
) -> np.ndarray:
    """固定长度的对称轮廓残差向量（像素）。

    长度为 ``2 * samples``：前半是"观测轮廓采样点 -> 投影轮廓边界"的距离，
    后半反过来。两个输入都会先取**凸包**——这一点很关键：``_project_cube`` 返回的是
    立方体 8 个投影顶点（其中 2 个落在轮廓内部），若直接按闭合折线处理会形成自交路径，
    把残差放大好几倍（实测把 1 px 的真实残差放大到 15 px，导致质量门限全部误判）。
    """
    observed_hull = cv2.convexHull(np.asarray(observed, dtype=np.float32)).reshape(-1, 2).astype(np.float64)
    projected_hull = cv2.convexHull(np.asarray(projected, dtype=np.float32)).reshape(-1, 2).astype(np.float64)
    if len(observed_hull) < 3 or len(projected_hull) < 3:
        return np.full(2 * samples, 1e3)

    observed_points = _resample_polygon(observed_hull, samples)
    projected_points = _resample_polygon(projected_hull, samples)
    forward = _point_to_polygon_distance(observed_points, projected_points)
    backward = _point_to_polygon_distance(projected_points, observed_points)
    if not (np.all(np.isfinite(forward)) and np.all(np.isfinite(backward))):
        return np.full(2 * samples, 1e3)
    return np.concatenate([forward, backward])


def _silhouette_distance(observed: np.ndarray, projected: np.ndarray) -> float:
    """对称轮廓距离（标量，像素）= 固定长度残差向量的 RMS。"""
    residuals = _silhouette_residuals(observed, projected)
    return float(math.sqrt(float(np.mean(residuals * residuals))))


def refine_cube_pose(
    observed_polygon: Sequence[Sequence[float]],
    edge_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    z_init: Optional[float] = None,
    rvec_init: Optional[Sequence[float]] = None,
    max_iterations: int = 12,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    """把已知边长的立方体拟合到观测轮廓，恢复 6DoF 位姿（单帧、无需人工对应）。

    **算法：循环对应 + PnP 迭代（ICP 思想）**，而不是直接对轮廓距离做梯度下降。

    为什么不用"点-轮廓距离 + LM"（踩过的坑）：那条路会因为最近线段归属跳变而陷入
    局部极小，实测轮廓残差只有 5 px 却带来 10~20% 的距离误差，且加迭代次数完全无改善。
    正确做法是**显式建立对应关系**：

    1. 由当前位姿投影立方体 8 顶点，取凸包得到投影轮廓顶点的**循环顺序**（并记下它们
       对应的立方体三维顶点）；
    2. 观测轮廓也是循环顺序，因此"对应关系"只差一个循环偏移和绕向——枚举
       ``2 × n`` 种组合，每种做一次 PnP，取重投影误差最小者；
    3. 用该位姿更新，重复若干次即收敛（通常 3~5 次迭代，误差降到 1 px 量级）。

    这样得到的位姿对任意视角成立，斜视（六边形轮廓）也能给出准确距离——
    而"正视 + 已知边长"的近似在斜视时有 ~25~30% 的系统误差（实测）。

    Args:
        observed_polygon: 观测轮廓点集（像素）。
        edge_m: 立方体边长（米，单目必须有它才有尺度）。
        camera_matrix: ``(3, 3)`` 内参。
        dist_coeffs: 畸变系数或 ``None``。
        z_init: 距离初值（米）；``None`` 时按轮廓面积估计（任意姿态下立方体轮廓面积
            期望约 ``1.4·L²``，比"最小外接矩形短边"稳健得多）。
        rvec_init: 旋转初值（Rodrigues）；``None`` 用零旋转。
        max_iterations: 最大迭代次数（默认 12，通常 5 次内收敛）。

    Returns:
        ``(ok, rvec, tvec, rms_px)``；``rms_px`` 为轮廓重投影残差，建议门限 2.5 px。
    """
    observed = np.asarray(observed_polygon, dtype=np.float64).reshape(-1, 2)
    if len(observed) < 4 or edge_m <= 0.0 or camera_matrix is None:
        return False, None, None, float("inf")

    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    focal = 0.5 * (float(camera_matrix[0, 0]) + float(camera_matrix[1, 1]))
    cube_points = _UNIT_CUBE * float(edge_m)

    # --- 初值 ---
    if z_init is None or not np.isfinite(z_init) or z_init <= 0.0:
        area = float(cv2.contourArea(observed.astype(np.float32)))
        # 任意姿态下立方体轮廓面积的期望约为 1.4·L²（正对 1.0，角对 √3≈1.73）
        z_init = max(1e-3, focal * float(edge_m) * math.sqrt(1.4 / max(area, 1.0)))
    center = observed.mean(axis=0)
    ray = pixel_to_ray(center[0], center[1], camera_matrix)
    rvec = np.zeros((3, 1), dtype=np.float64) if rvec_init is None else np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
    tvec = (ray * float(z_init)).reshape(3, 1)

    observed_hull = cv2.convexHull(observed.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    n_observed = len(observed_hull)
    if n_observed < 3:
        return False, None, None, float("inf")

    best_state: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

    # 先用"零旋转"单起点：正视/近正视视角这一步就能收敛，因此不需要付多起点的代价
    # （实测：多起点会把正视这个本来很准的解带到错误盆地，误差从 0.4% 恶化到 23%）
    fast = _icp_cube_fit(
        observed_hull, cube_points, camera_matrix, dist,
        np.zeros(3) if rvec_init is None else np.asarray(rvec_init, dtype=np.float64).reshape(3),
        ray * float(z_init), max(4, int(max_iterations)),
    )
    if fast is not None:
        fast_rms = _silhouette_distance(
            observed, _project_cube(fast[1], fast[2], edge_m, camera_matrix, dist)
        )
        if np.isfinite(fast_rms) and fast_rms <= 2.0:
            return True, fast[1].reshape(3).copy(), fast[2].reshape(3).copy(), float(fast_rms)
        best_state = fast

    # 单起点不达标（典型情况：斜视六边形轮廓）时，用立方体对称性构造多个粗初值再精修
    for seed_rotation in _rotation_seeds():
        state = _icp_cube_fit(
            observed_hull, cube_points, camera_matrix, dist, seed_rotation, ray * float(z_init), 3,
        )
        if state is None:
            continue
        error = _silhouette_distance(observed, _project_cube(state[1], state[2], edge_m, camera_matrix, dist))
        if not np.isfinite(error):
            continue
        if best_state is None or error < _silhouette_distance(
            observed, _project_cube(best_state[1], best_state[2], edge_m, camera_matrix, dist)
        ):
            best_state = state

    if best_state is None:
        return False, None, None, float("inf")

    refined = _icp_cube_fit(
        observed_hull, cube_points, camera_matrix, dist, best_state[1], best_state[2],
        max(4, int(max_iterations)),
    )
    if refined is not None:
        refined_rms = _silhouette_distance(
            observed, _project_cube(refined[1], refined[2], edge_m, camera_matrix, dist)
        )
        if np.isfinite(refined_rms):
            best_state = refined

    rvec, tvec = best_state[1], best_state[2]
    final_projected = _project_cube(rvec, tvec, edge_m, camera_matrix, dist)
    rms = _silhouette_distance(observed, final_projected)
    if not np.isfinite(rms) or float(np.asarray(tvec).reshape(-1)[2]) <= 1e-6:
        return False, None, None, float("inf")
    return True, rvec.reshape(3).copy(), tvec.reshape(3).copy(), float(rms)


def _rotation_seeds() -> List[np.ndarray]:
    """多起点旋转初值：立方体对称性 + 若干斜视方向。

    这些初值覆盖了"正对某个面""斜看棱""角朝相机"等典型视角，够用且数量少（9 个），
    因为真正的收敛靠后面的 ICP 迭代，初值只需要把解带进正确的盆地。
    """
    seeds: List[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    quarter = math.pi / 4.0
    for rx in (0.0, quarter, -quarter):
        for ry in (0.0, quarter, -quarter):
            if rx == 0.0 and ry == 0.0:
                continue
            seeds.append(np.array([rx, ry, 0.0], dtype=np.float64))
    seeds.append(np.array([quarter, quarter, quarter], dtype=np.float64))
    return seeds


def _alignment_score(projected_subset: np.ndarray, image_subset: np.ndarray) -> float:
    """循环对应候选的廉价评分：归一化（去中心、除以尺度）后对应点的平方距离均值。

    只用于挑出"最可能正确的那个循环偏移"，真正的位姿由随后的 PnP 求解。
    """
    if len(projected_subset) != len(image_subset) or len(projected_subset) == 0:
        return float("inf")
    a = np.asarray(projected_subset, dtype=np.float64)
    b = np.asarray(image_subset, dtype=np.float64)
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    a_scale = float(np.sqrt(np.mean(np.sum(a_centered * a_centered, axis=1)))) + 1e-9
    b_scale = float(np.sqrt(np.mean(np.sum(b_centered * b_centered, axis=1)))) + 1e-9
    delta = a_centered / a_scale - b_centered / b_scale
    return float(np.mean(np.sum(delta * delta, axis=1)))


def _icp_cube_fit(
    observed_hull: np.ndarray,
    cube_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    iterations: int,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """一轮"循环对应 + PnP"的立方体位姿迭代（ICP 思想的单次调用）。

    Args:
        observed_hull: 观测轮廓凸包顶点（循环顺序，像素）。
        cube_points: 立方体 8 个三维顶点（米）。
        camera_matrix: 内参。
        dist: 畸变系数 ``(1, N)``。
        rvec_init: 旋转初值 ``(3,)`` 或 ``(3, 1)``。
        tvec_init: 平移初值（相机坐标系）。
        iterations: 迭代次数。

    Returns:
        ``(reprojection_error_px, rvec(3,), tvec(3,))``；全部候选对应都失败时返回 ``None``。
    """
    rvec = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1).copy()
    tvec = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1).copy()
    n_observed = len(observed_hull)
    best_error = float("inf")

    for _ in range(max(1, int(iterations))):
        projected_all, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist)
        projected_all = projected_all.reshape(8, 2)
        if not np.all(np.isfinite(projected_all)):
            return None

        hull_indices = cv2.convexHull(projected_all.astype(np.float32), returnPoints=False).ravel()
        if len(hull_indices) < 3:
            return None

        hull_object = cube_points[hull_indices]
        pairs = min(len(hull_indices), n_observed)
        if pairs < 4:
            return None

        step_best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        projected_hull = projected_all[hull_indices]

        # 先把 2×n 种循环对应做一次**廉价几何评分**（归一化后对应点距离），
        # 只对最优的两个候选跑 PnP。若对每个候选都跑 PnP，单帧耗时会到 90~150 ms，
        # 实测这样改后降到 15 ms 量级且精度不变。
        candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for direction in (1, -1):
            ordered_object = hull_object[::direction]
            ordered_projected = projected_hull[::direction]
            ordered_image = observed_hull if direction == 1 else observed_hull[::-1]
            for shift in range(pairs):
                object_subset = np.roll(ordered_object, -shift, axis=0)[:pairs]
                projected_subset = np.roll(ordered_projected, -shift, axis=0)[:pairs]
                image_subset = ordered_image[:pairs]
                candidates.append((_alignment_score(projected_subset, image_subset), object_subset, image_subset))
        candidates.sort(key=lambda item: item[0])

        for _score, object_subset, image_subset in candidates[:2]:
            try:
                ok, rvec_candidate, tvec_candidate = cv2.solvePnP(
                    object_subset.reshape(pairs, 1, 3).astype(np.float64),
                    image_subset.reshape(pairs, 1, 2).astype(np.float64),
                    camera_matrix,
                    dist,
                    rvec.copy(),
                    tvec.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                continue
            if not ok or tvec_candidate is None:
                continue
            tvec_flat = np.asarray(tvec_candidate, dtype=np.float64).reshape(3)
            if tvec_flat[2] <= 1e-6:
                continue
            reproj, _ = cv2.projectPoints(object_subset, rvec_candidate, tvec_candidate, camera_matrix, dist)
            residual = reproj.reshape(pairs, 2) - image_subset
            error = float(math.sqrt(float(np.mean(np.sum(residual * residual, axis=1)))))
            if step_best is None or error < step_best[0]:
                step_best = (error, np.asarray(rvec_candidate, dtype=np.float64).reshape(3, 1), tvec_flat.reshape(3, 1))

        if step_best is None:
            return None

        error, rvec_new, tvec_new = step_best
        movement = float(np.linalg.norm(rvec_new - rvec) + np.linalg.norm(tvec_new - tvec))
        rvec, tvec = rvec_new, tvec_new
        best_error = error
        if error <= 0.3 or movement < 1e-7:
            break

    if not np.isfinite(best_error):
        return None
    return best_error, rvec.reshape(3).copy(), tvec.reshape(3).copy()
