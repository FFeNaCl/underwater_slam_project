"""几何旁路立方体检测（不依赖任何训练模型）。

设计动机：YOLOv8 数据集尚未就绪，但**立方体是可以被几何严格刻画的物体**，
因此先用"颜色分割 + 轮廓几何 + 棱线结构"三条证据把它检出来，让整条链路
（增强 -> 检测 -> SLAM -> 语义地标 -> RViz）今天就能跑通。等数据集好了，
把 YOLO 节点挂到同一条 `/detections` 话题上即可无缝替换（消息契约见 core/types.py）。

立方体的几何事实（这是本文件所有判据的来源）：

* 从任意视角看，立方体的**外轮廓是凸多边形**，顶点数 = 可见面数 + 3，
  即 1 面 -> 4 顶点（正方形）、2 面 -> 5 顶点、3 面 -> 6 顶点（正六边形）。
* 轮廓面积 / 最小外接矩形面积（``fill_ratio``）：正方形 = 1.0，正六边形 ≈ 0.87。
  细长形状、残缺形状、反光条带都会明显更低。
* 三面视角时轮廓**内部**会出现 Y 型顶点与三条棱 -> 内部直线方向数 ≈ 3；
  而一块平面色斑（贴纸、反光、瓷砖）内部没有直线结构 -> 方向数 ≈ 0。
  这是"真立方体 vs 红色平面"最关键的区分手段，见 :func:`interior_edge_analysis`。

局限（务必如实写在报告里）：
* 强浑浊/强散射下颜色分割会碎，此时应打开 ``use_interior_edge_check`` 并降低
  ``min_convexity``，或改走 YOLO；
* 立方体完全背光呈黑块时颜色证据失效（可用 ``color_preset: custom`` 改为灰度阈值）；
* 与相机平面几乎共面的单面视角（只有 4 顶点、无内部棱）与任意正方形色块原理上
  不可区分，这是单目 + 无纹理物体的固有限制，不承诺解决。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import colors
from .cv_compat import find_contours, to_gray
from .measurement import measure_geometry
from .params import CubeDetectorConfig
from .tracker import IoUTracker
from .types import Detection, GeometryMetrics, Measurement, polygon_iou, xyxy_from_polygon

__all__ = [
    "CubeObservation",
    "interior_edge_analysis",
    "polygon_metrics",
    "detect_observations",
    "CubeDetector",
]


@dataclass
class CubeObservation:
    """一个候选立方体（几何层面，尚未跟踪）。"""

    detection: Detection
    polygon: np.ndarray
    metrics: GeometryMetrics
    score_terms: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return float(self.detection.confidence)


def _odd_kernel(value: int) -> int:
    """核大小规范为奇数，<=1 视为关闭。"""
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def interior_edge_analysis(
    gray: np.ndarray,
    polygon: np.ndarray,
    area_px: float,
    config: CubeDetectorConfig,
) -> Tuple[int, int, float, float]:
    """分析轮廓**内部**的直线结构（区分"真立方体"与"平面色块"的关键）。

    步骤：把轮廓向内腐蚀掉边界 -> 只保留内部边缘 -> Canny + 概率霍夫 ->
    按角度（模 180°）聚类统计独立方向数 -> 统计内部直线总长占比 ->
    Harris 角点响应峰值（三面视角的 Y 型顶点会给出强响应）。

    Args:
        gray: 单通道灰度图（与 polygon 同一坐标系）。
        polygon: 轮廓点集，形状 ``(N, 2)``。
        area_px: 轮廓面积（像素²），用于自适应线长门限。
        config: 检测参数。

    Returns:
        ``(direction_count, line_count, edge_support, corner_response)``，
        其中 ``edge_support`` = 内部直线总长 / 轮廓周长（0~1），
        ``corner_response`` 为按全图峰值归一化后的 Harris 响应（0~1）。
    """
    det_cfg = config.detect
    empty = (0, 0, 0.0, 0.0)
    if len(polygon) < 3 or area_px <= 1.0:
        return empty

    height, width = gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)

    # 向内腐蚀，排除轮廓自身的边界（否则量到的是外轮廓而不是内部棱线）
    erode_size = max(3, int(round(0.06 * math.sqrt(area_px))))
    erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)))
    if int(np.count_nonzero(inner)) < 16:
        return empty

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, int(det_cfg.canny_low), int(det_cfg.canny_high))
    edges_inner = cv2.bitwise_and(edges, inner)

    perimeter = float(cv2.arcLength(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2), True))
    min_len = max(8, int(det_cfg.min_edge_length_frac * math.sqrt(area_px)))
    lines = cv2.HoughLinesP(
        edges_inner,
        rho=1,
        theta=math.pi / 180.0,
        threshold=max(10, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(3, min_len // 3),
    )

    direction_angles: List[float] = []
    total_length = 0.0
    line_count = 0
    if lines is not None:
        for raw in lines:
            x1, y1, x2, y2 = (float(v) for v in np.asarray(raw).reshape(4))
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_len:
                continue
            line_count += 1
            total_length += length
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            for index, existing in enumerate(direction_angles):
                delta = abs(angle - existing)
                delta = min(delta, 180.0 - delta)
                if delta <= det_cfg.angle_cluster_tolerance_deg:
                    # 用长度加权更新簇中心，避免长线被短线带偏
                    direction_angles[index] = existing  # 保持首个代表值，聚类只做计数
                    break
            else:
                direction_angles.append(angle)

    edge_support = float(total_length / perimeter) if perimeter > 1e-6 else 0.0

    # Harris 角点响应：三面视角的 Y 型顶点响应显著高于平面色斑
    corner_response = 0.0
    gray_f32 = np.float32(blurred)
    harris = cv2.cornerHarris(gray_f32, blockSize=3, ksize=3, k=0.04)
    global_peak = float(np.max(harris)) if harris.size else 0.0
    if global_peak > 1e-9:
        inner_harris = np.where(inner > 0, harris, 0.0)
        corner_response = float(np.max(inner_harris) / global_peak)
    corner_response = max(0.0, min(1.0, corner_response))

    return len(direction_angles), line_count, edge_support, corner_response


def polygon_metrics(
    contour: np.ndarray,
    polygon: np.ndarray,
    area_px: float,
) -> GeometryMetrics:
    """计算轮廓/多边形的几何指标（对应 :class:`GeometryMetrics` 中除棱线与颜色外的字段）。"""
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    rect = cv2.minAreaRect(contour)
    (_, _), (rect_w, rect_h), angle = rect
    rect_area = float(max(rect_w * rect_h, 1e-6))
    short_side = float(min(rect_w, rect_h))
    long_side = float(max(rect_w, rect_h))

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = float(max(bw * bh, 1))

    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
    else:
        center = (float(x + bw / 2.0), float(y + bh / 2.0))

    return GeometryMetrics(
        center=center,
        corners=[[float(p[0]), float(p[1])] for p in np.asarray(polygon).reshape(-1, 2)],
        vertex_count=int(len(polygon)),
        area_px=float(area_px),
        hull_area_px=hull_area,
        fill_ratio=float(area_px / rect_area),
        convexity=float(area_px / hull_area) if hull_area > 1e-6 else 0.0,
        extent=float(area_px / bbox_area),
        aspect_ratio=float(short_side / long_side) if long_side > 1e-6 else 0.0,
        angle_deg=float(angle),
        side_px=short_side,
        long_side_px=long_side,
    )


def _score(
    metrics: GeometryMetrics,
    color_purity: float,
    config: CubeDetectorConfig,
) -> Tuple[float, Dict[str, float]]:
    """把各项指标折算成 0~1 的分数并加权求和。

    Returns:
        ``(confidence, terms)``；``terms`` 会写进消息里，便于现场定位"为什么没检到"。
    """
    det = config.detect

    def ramp(value: float, low: float, high: float = 1.0) -> float:
        if high <= low:
            return 1.0 if value >= high else 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    terms: Dict[str, float] = {}
    terms["fill"] = ramp(metrics.fill_ratio, det.min_fill_ratio, 0.95)
    terms["convexity"] = ramp(metrics.convexity, det.min_convexity, 0.99)
    terms["aspect"] = ramp(metrics.aspect_ratio, det.min_aspect_ratio, 0.9)

    if det.use_interior_edge_check:
        if det.mode == "square_face":
            terms["edge"] = 1.0 if metrics.vertex_count == 4 else 0.0
        else:
            direction_score = ramp(float(metrics.edge_direction_count), float(det.min_edge_directions), 3.0)
            support_score = ramp(metrics.edge_support, det.min_edge_support, det.min_edge_support * 3.0 + 1e-6)
            corner_score = ramp(
                metrics.internal_corner_response,
                det.min_internal_corner_response,
                min(0.6, det.min_internal_corner_response * 4.0 + 1e-6),
            )
            terms["edge"] = 0.5 * direction_score + 0.3 * support_score + 0.2 * corner_score
    else:
        terms["edge"] = 1.0

    terms["color"] = max(0.0, min(1.0, color_purity))

    # 面积项：目标太大或太小都不可信；以 preferred_side_px 为中心做对数高斯
    if det.preferred_side_px > 0 and metrics.side_px > 0:
        sigma = 0.9
        ratio = math.log(max(metrics.side_px, 1e-3) / det.preferred_side_px)
        terms["area"] = float(math.exp(-0.5 * (ratio / sigma) ** 2))
    else:
        terms["area"] = 1.0

    weights = {
        "fill": det.weight_fill,
        "convexity": det.weight_convexity,
        "aspect": det.weight_aspect,
        "edge": det.weight_edge,
        "color": det.weight_color,
        "area": det.weight_area,
    }
    total_weight = sum(max(0.0, w) for w in weights.values())
    if total_weight <= 0.0:
        return 0.0, terms
    confidence = sum(max(0.0, weights[k]) * terms[k] for k in weights) / total_weight
    return float(max(0.0, min(1.0, confidence))), terms


def _mode_gate(metrics: GeometryMetrics, config: CubeDetectorConfig) -> bool:
    """按 ``mode`` 做硬性几何门限。"""
    mode = (config.detect.mode or "any").lower()
    if mode == "square_face":
        return metrics.vertex_count == 4
    if mode == "cube_silhouette":
        if metrics.vertex_count >= 5:
            return True
        return metrics.edge_direction_count >= config.detect.min_edge_directions
    return True


def detect_observations(
    image: np.ndarray,
    config: CubeDetectorConfig,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    return_mask: bool = False,
):
    """在单帧图像上做几何检测（不含跟踪）。

    Args:
        image: BGR 图像（``uint8``）。
        config: 检测配置。
        camera_matrix: 相机内参；``None`` 时用配置里的（可能也没有 -> 只输出像素信息）。
        dist_coeffs: 畸变系数；``None`` 时用配置里的。
        return_mask: 是否一并返回颜色掩码（调试/CLI 用）。

    Returns:
        ``List[CubeObservation]``；``return_mask=True`` 时返回 ``(observations, mask)``，
        掩码已经还原到**原图**尺寸，方便叠加显示。
    """
    det_cfg = config.detect
    if image is None or image.size == 0:
        return ([], None) if return_mask else []

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # --- ROI 裁剪（水底反光带常干扰，裁掉最快） ---
    offset_x, offset_y = 0, 0
    work = image
    roi = list(det_cfg.roi)
    if len(roi) == 4 and roi[2] > 0 and roi[3] > 0:
        x, y, w, h = (int(v) for v in roi)
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = max(1, min(w, image.shape[1] - x))
        h = max(1, min(h, image.shape[0] - y))
        work = image[y : y + h, x : x + w]
        offset_x, offset_y = x, y

    # --- 缩放提速（结果坐标最后还原） ---
    scale = 1.0
    if det_cfg.detect_width > 0 and work.shape[1] > det_cfg.detect_width:
        scale = float(det_cfg.detect_width) / float(work.shape[1])
        work = cv2.resize(work, (int(round(work.shape[1] * scale)), int(round(work.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    mask = colors.build_color_mask(work, config.color)
    gray = to_gray(work)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    work_area = float(work.shape[0] * work.shape[1])
    max_area = det_cfg.max_area_fraction * work_area

    contours, _ = find_contours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if camera_matrix is None:
        camera_matrix = config.measure.camera_matrix()
    if dist_coeffs is None:
        dist_coeffs = config.measure.distortion()

    observations: List[CubeObservation] = []
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        if area_px < det_cfg.min_area_px or area_px > max_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        polygon = cv2.approxPolyDP(contour, det_cfg.approx_epsilon_ratio * perimeter, True)
        vertex_count = int(len(polygon))
        if vertex_count < det_cfg.min_vertices or vertex_count > det_cfg.max_vertices:
            continue
        if not cv2.isContourConvex(polygon):
            continue

        metrics = polygon_metrics(contour, polygon, area_px)
        if metrics.fill_ratio < det_cfg.min_fill_ratio:
            continue
        if metrics.convexity < det_cfg.min_convexity:
            continue
        if metrics.aspect_ratio < det_cfg.min_aspect_ratio:
            continue

        if det_cfg.use_interior_edge_check:
            directions, line_count, support, corner = interior_edge_analysis(gray, polygon, area_px, config)
            metrics.edge_direction_count = directions
            metrics.interior_edge_count = line_count
            metrics.edge_support = support
            metrics.internal_corner_response = corner

        purity, mean_hue, mean_sat, mean_val = colors.mask_statistics(
            hsv, mask, np.asarray(polygon).reshape(-1, 2)
        )
        metrics.color_purity = purity
        metrics.mean_hue = mean_hue
        metrics.mean_saturation = mean_sat
        metrics.mean_value = mean_val

        if not _mode_gate(metrics, config):
            continue

        confidence, terms = _score(metrics, purity, config)
        if confidence < det_cfg.min_confidence:
            continue

        # --- 坐标还原到原图尺度 ---
        restore = np.asarray(metrics.corners, dtype=np.float64).reshape(-1, 2)
        restore = restore / scale
        restore[:, 0] += offset_x
        restore[:, 1] += offset_y
        metrics.corners = [[float(p[0]), float(p[1])] for p in restore]
        metrics.center = (
            float(metrics.center[0] / scale + offset_x),
            float(metrics.center[1] / scale + offset_y),
        )
        metrics.area_px = float(metrics.area_px / (scale * scale))
        metrics.hull_area_px = float(metrics.hull_area_px / (scale * scale))
        metrics.side_px = float(metrics.side_px / scale)
        metrics.long_side_px = float(metrics.long_side_px / scale)

        xyxy = xyxy_from_polygon(metrics.corners)
        detection = Detection(
            class_id=config.class_id,
            class_name=config.class_name,
            confidence=confidence,
            xyxy=xyxy,
            score_terms=terms,
            geometry=metrics,
        )

        # --- 单目测量（射线必有；距离需内参 + 已知边长） ---
        if config.measure.enabled:
            result = measure_geometry(
                metrics,
                confidence,
                camera_matrix,
                dist_coeffs,
                edge_m=config.measure.cube_edge_m,
                polygon=metrics.corners,
                min_side_px=config.measure.min_side_px_for_distance,
                use_silhouette_fit=config.measure.use_silhouette_fit,
                fit_max_rms_px=config.measure.fit_max_rms_px,
            )
            measurement = result.to_measurement()
            if measurement.method != "none" or measurement.ray_camera is not None:
                detection.measurement = measurement

        observations.append(CubeObservation(detection=detection, polygon=np.asarray(metrics.corners), metrics=metrics, score_terms=terms))

    observations = _nms(observations, det_cfg.nms_iou, det_cfg.max_detections)

    if not return_mask:
        return observations

    full_mask = mask
    if scale != 1.0 or offset_x or offset_y:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        resized = cv2.resize(mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST)
        full_mask[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return observations, full_mask


def _nms(observations: List[CubeObservation], iou_threshold: float, max_detections: int) -> List[CubeObservation]:
    """按置信度做多边形 IoU 非极大值抑制（同一立方体常被分割成多个候选）。"""
    if not observations:
        return []
    ordered = sorted(observations, key=lambda o: o.confidence, reverse=True)
    kept: List[CubeObservation] = []
    for candidate in ordered:
        if max_detections > 0 and len(kept) >= max_detections:
            break
        redundant = False
        for accepted in kept:
            if polygon_iou(candidate.metrics.corners, accepted.metrics.corners) > iou_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(candidate)
    return kept


class CubeDetector:
    """有状态的检测器：检测 + 跟踪 + 测量（ROS 节点与 CLI 共用的门面）。

    Example:
        >>> detector = CubeDetector(CubeDetectorConfig.from_flat(load_flat_yaml(path)))
        >>> detections = detector.detect(bgr_image)      # 已带 track_id
        >>> detector.last_mask.shape                     # 调试用掩码
    """

    def __init__(self, config: Optional[CubeDetectorConfig] = None) -> None:
        self.config = config or CubeDetectorConfig()
        self.tracker = IoUTracker(self.config.track)
        self.last_mask: Optional[np.ndarray] = None
        self.last_observations: List[CubeObservation] = []
        self._last_stamp: Optional[float] = None

    def reset(self) -> None:
        """重置跟踪状态（换场景 / SLAM 重定位后调用）。"""
        self.tracker.reset()
        self._last_stamp = None

    def detect(self, image: np.ndarray, stamp: Optional[float] = None) -> List[Detection]:
        """检测一帧并返回带 ``track_id`` 的检测结果。

        Args:
            image: BGR 图像。
            stamp: 可选的时间戳（秒，仅用于限频判断，可传 ``None``）。

        Returns:
            已确认（默认 ``min_hits=2``）的检测列表。
        """
        observations, mask = detect_observations(image, self.config, return_mask=True)
        self.last_observations = observations
        self.last_mask = mask
        self._last_stamp = stamp
        detections = [obs.detection for obs in observations]
        return self.tracker.update(detections)
