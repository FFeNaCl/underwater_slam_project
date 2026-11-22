"""检测消息数据契约（schema_version = 1）。

设计约束（改动前请先读）：
1. **向后兼容**：现有 ``yolo_vrx_bridge/yolo_detector_node.py`` 发布
   ``/yolo/detections``（``std_msgs/String``，JSON）。本模块产生的 JSON 是它的
   *超集*：``stamp`` / ``frame_id`` / ``source`` / ``count`` / ``detections`` 五个字段
   语义完全一致，``detections[i]`` 至少含 ``class_id`` / ``class_name`` /
   ``confidence`` / ``xyxy``。因此任何"只认老字段"的下游代码无需修改即可消费几何旁路结果。
2. **新增字段只增不改**：几何与测量信息放在 ``geometry`` / ``measurement`` 子字典里，
   不要往 detection 顶层塞新字段，避免与别的检测器（YOLO）冲突。
3. **坐标约定**：所有像素坐标都是 ``(u, v)``，``u`` 向右、``v`` 向下，原点在图像左上角；
   图像尺寸在 ``image.width`` / ``image.height`` 给出，方便下游做归一化。
4. **时间戳**：``stamp.sec`` + ``stamp.nanosec``，与 ROS 消息头一致；没有 ROS 头时
   由调用方填 0，下游必须容忍 0。

下游消费方（语义地标节点、RViz、离线评估脚本）请只依赖本文件定义的字段。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

#: 几何旁路检测器的 source 标识（YOLO 节点用 "model" / "label_replay"）
SOURCE_GEOMETRY = "geometry"

__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_GEOMETRY",
    "GeometryMetrics",
    "Measurement",
    "Detection",
    "DetectionFrame",
    "xyxy_from_polygon",
    "polygon_area",
    "polygon_iou",
    "bbox_iou",
]


def _as_float_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def _as_point_list(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(p[0]), float(p[1])] for p in points]


@dataclass
class GeometryMetrics:
    """几何验证指标：既能给人看，也是区分"真立方体 / 反光斑 / 色块"的依据。"""

    center: Tuple[float, float]
    corners: List[List[float]] = field(default_factory=list)
    vertex_count: int = 0
    area_px: float = 0.0
    hull_area_px: float = 0.0
    fill_ratio: float = 0.0            # 轮廓面积 / 最小外接矩形面积（正方形=1，正六边形≈0.87）
    convexity: float = 0.0             # 轮廓面积 / 凸包面积（凸形状=1）
    extent: float = 0.0                # 轮廓面积 / 正外接矩形面积
    aspect_ratio: float = 0.0          # 短边 / 长边（外接矩形），正视面接近 1
    angle_deg: float = 0.0             # 最小外接矩形角度，用于判断倾斜
    side_px: float = 0.0               # 外接矩形短边长（像素），已知边长测距用
    long_side_px: float = 0.0          # 外接矩形长边长（像素）
    edge_direction_count: int = 0      # 轮廓内部检测到的独立直线方向数（立方体三面视角=3）
    interior_edge_count: int = 0       # 轮廓内部直线段数量
    edge_support: float = 0.0          # 内部直线总长 / 轮廓周长
    internal_corner_response: float = 0.0  # 内部角点（Y 型顶点）最强 Harris 响应，归一化
    color_purity: float = 0.0          # 轮廓内属于目标颜色的像素占比
    mean_hue: float = 0.0
    mean_saturation: float = 0.0
    mean_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": _as_float_list(self.center),
            "corners": _as_point_list(self.corners),
            "vertex_count": int(self.vertex_count),
            "area_px": float(self.area_px),
            "hull_area_px": float(self.hull_area_px),
            "fill_ratio": float(self.fill_ratio),
            "convexity": float(self.convexity),
            "extent": float(self.extent),
            "aspect_ratio": float(self.aspect_ratio),
            "angle_deg": float(self.angle_deg),
            "side_px": float(self.side_px),
            "long_side_px": float(self.long_side_px),
            "edge_direction_count": int(self.edge_direction_count),
            "interior_edge_count": int(self.interior_edge_count),
            "edge_support": float(self.edge_support),
            "internal_corner_response": float(self.internal_corner_response),
            "color_purity": float(self.color_purity),
            "mean_hue": float(self.mean_hue),
            "mean_saturation": float(self.mean_saturation),
            "mean_value": float(self.mean_value),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GeometryMetrics":
        center = data.get("center", (0.0, 0.0))
        return GeometryMetrics(
            center=(float(center[0]), float(center[1])),
            corners=_as_point_list(data.get("corners", [])),
            vertex_count=int(data.get("vertex_count", 0)),
            area_px=float(data.get("area_px", 0.0)),
            hull_area_px=float(data.get("hull_area_px", 0.0)),
            fill_ratio=float(data.get("fill_ratio", 0.0)),
            convexity=float(data.get("convexity", 0.0)),
            extent=float(data.get("extent", 0.0)),
            aspect_ratio=float(data.get("aspect_ratio", 0.0)),
            angle_deg=float(data.get("angle_deg", 0.0)),
            side_px=float(data.get("side_px", 0.0)),
            long_side_px=float(data.get("long_side_px", 0.0)),
            edge_direction_count=int(data.get("edge_direction_count", 0)),
            interior_edge_count=int(data.get("interior_edge_count", 0)),
            edge_support=float(data.get("edge_support", 0.0)),
            internal_corner_response=float(data.get("internal_corner_response", 0.0)),
            color_purity=float(data.get("color_purity", 0.0)),
            mean_hue=float(data.get("mean_hue", 0.0)),
            mean_saturation=float(data.get("mean_saturation", 0.0)),
            mean_value=float(data.get("mean_value", 0.0)),
        )


@dataclass
class Measurement:
    """单目测量结果。**注意**：单目无绝对尺度，``distance_m`` 只在
    ``method == "known_size"``（已知立方体边长）或 ``method == "pose"``（由 SLAM 位姿
    与多视图三角化给出）时才有物理意义，其余情况只是像素级估计。"""

    distance_m: Optional[float] = None            # 相机光心到目标中心的距离（米）
    bearing_rad: Optional[Tuple[float, float]] = None  # 相机坐标系下的方位角 (方位, 俯仰)
    ray_camera: Optional[List[float]] = None      # 相机坐标系下的单位射线 (x, y, z)
    method: str = "none"                          # none | bearing_only | known_size | known_size_pose | known_size_refined
    quality: float = 0.0                          # 0~1，随目标尺寸/遮挡/畸变下降
    #: 由"已知边长 + 轮廓拟合"得到的立方体位姿（相机坐标系），可用于 RViz 画朝向
    pose_rvec: Optional[List[float]] = None       # Rodrigues 旋转（物体->相机）
    pose_tvec: Optional[List[float]] = None       # 平移（相机坐标系，米）
    fit_error_px: Optional[float] = None          # 轮廓拟合残差（像素），越大越不可信

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"method": self.method, "quality": float(self.quality)}
        if self.distance_m is not None:
            out["distance_m"] = float(self.distance_m)
        if self.bearing_rad is not None:
            out["bearing_rad"] = _as_float_list(self.bearing_rad)
        if self.ray_camera is not None:
            out["ray_camera"] = _as_float_list(self.ray_camera)
        if self.pose_rvec is not None:
            out["pose_rvec"] = _as_float_list(self.pose_rvec)
        if self.pose_tvec is not None:
            out["pose_tvec"] = _as_float_list(self.pose_tvec)
        if self.fit_error_px is not None:
            out["fit_error_px"] = float(self.fit_error_px)
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Measurement":
        bearing = data.get("bearing_rad")
        ray = data.get("ray_camera")
        rvec = data.get("pose_rvec")
        tvec = data.get("pose_tvec")
        return Measurement(
            distance_m=None if data.get("distance_m") is None else float(data["distance_m"]),
            bearing_rad=None if bearing is None else (float(bearing[0]), float(bearing[1])),
            ray_camera=None if ray is None else _as_float_list(ray),
            method=str(data.get("method", "none")),
            quality=float(data.get("quality", 0.0)),
            pose_rvec=None if rvec is None else _as_float_list(rvec),
            pose_tvec=None if tvec is None else _as_float_list(tvec),
            fit_error_px=None if data.get("fit_error_px") is None else float(data["fit_error_px"]),
        )


@dataclass
class Detection:
    """单个目标检测结果。

    Attributes:
        class_id: 类别编号（几何旁路默认 0，单类别）。
        class_name: 类别名（默认 ``cube``）。
        confidence: 0~1 置信度。几何旁路中是几何+颜色证据的加权评分，不是模型概率。
        xyxy: ``[x1, y1, x2, y2]`` 像素外接框（与 YOLO 节点字段完全一致）。
        score_terms: 评分各分量，便于现场调参定位问题。
        track_id: 跨帧跟踪 ID，未启用跟踪时为 ``None``。
        geometry: 几何指标。
        measurement: 单目测量结果。
        detector: 产生该检测的检测器标识（``geometry`` / ``model`` / ``label_replay``）。
    """

    class_id: int
    class_name: str
    confidence: float
    xyxy: List[float]
    score_terms: Dict[str, float] = field(default_factory=dict)
    track_id: Optional[int] = None
    geometry: Optional[GeometryMetrics] = None
    measurement: Optional[Measurement] = None
    detector: str = SOURCE_GEOMETRY

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": float(self.confidence),
            "xyxy": _as_float_list(self.xyxy),
        }
        if self.track_id is not None:
            out["track_id"] = int(self.track_id)
        if self.detector:
            out["detector"] = str(self.detector)
        if self.score_terms:
            out["score_terms"] = {k: float(v) for k, v in self.score_terms.items()}
        if self.geometry is not None:
            out["geometry"] = self.geometry.to_dict()
        if self.measurement is not None:
            out["measurement"] = self.measurement.to_dict()
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Detection":
        geometry = data.get("geometry")
        measurement = data.get("measurement")
        return Detection(
            class_id=int(data.get("class_id", 0)),
            class_name=str(data.get("class_name", "object")),
            confidence=float(data.get("confidence", 0.0)),
            xyxy=_as_float_list(data.get("xyxy", [0.0, 0.0, 0.0, 0.0])),
            score_terms={k: float(v) for k, v in (data.get("score_terms") or {}).items()},
            track_id=None if data.get("track_id") is None else int(data["track_id"]),
            geometry=None if geometry is None else GeometryMetrics.from_dict(geometry),
            measurement=None if measurement is None else Measurement.from_dict(measurement),
            detector=str(data.get("detector", data.get("source", SOURCE_GEOMETRY))),
        )

    @property
    def center(self) -> Tuple[float, float]:
        """外接框中心（像素）。"""
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def width(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    def polygon(self) -> List[List[float]]:
        """返回几何轮廓（若有），否则退化为外接框四角。"""
        if self.geometry is not None and len(self.geometry.corners) >= 3:
            return [list(p) for p in self.geometry.corners]
        x1, y1, x2, y2 = self.xyxy
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


@dataclass
class DetectionFrame:
    """一帧的完整检测结果，序列化后即为 ROS 话题上的 JSON。"""

    detections: List[Detection] = field(default_factory=list)
    stamp_sec: int = 0
    stamp_nanosec: int = 0
    frame_id: str = ""
    image_width: int = 0
    image_height: int = 0
    source: str = SOURCE_GEOMETRY
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stamp": {"sec": int(self.stamp_sec), "nanosec": int(self.stamp_nanosec)},
            "frame_id": str(self.frame_id),
            "source": str(self.source),
            "count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }
        if self.image_width and self.image_height:
            out["image"] = {"width": int(self.image_width), "height": int(self.image_height)}
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    def to_json(self) -> str:
        """序列化为单行 JSON（``ensure_ascii`` 打开，避免非 ASCII 在 ROS 日志里乱码）。"""
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DetectionFrame":
        stamp = data.get("stamp") or {}
        image = data.get("image") or {}
        return DetectionFrame(
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
            stamp_sec=int(stamp.get("sec", 0)),
            stamp_nanosec=int(stamp.get("nanosec", 0)),
            frame_id=str(data.get("frame_id", "")),
            image_width=int(image.get("width", 0)),
            image_height=int(image.get("height", 0)),
            source=str(data.get("source", SOURCE_GEOMETRY)),
            meta=dict(data.get("meta") or {}),
        )

    @staticmethod
    def from_json(text: str) -> "DetectionFrame":
        """解析 JSON 文本；兼容旧 YOLO 节点的输出（缺 schema_version / image 也能解析）。"""
        return DetectionFrame.from_dict(json.loads(text))


def xyxy_from_polygon(points: Sequence[Sequence[float]]) -> List[float]:
    """由多边形点集计算外接框 ``[x1, y1, x2, y2]``。"""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """鞋带公式计算多边形面积（像素²，返回绝对值）。"""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 ``xyxy`` 外接框的 IoU。"""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def polygon_iou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """两个凸多边形的 IoU（用轮廓掩码近似，避免依赖 shapely）。

    实现方式：把两个多边形栅格化到同一张小掩码上求交并集。对跟踪关联足够精确，
    且完全依赖 numpy，便于离线测试。
    """
    if len(a) < 3 or len(b) < 3:
        return 0.0
    pts_a = [(float(p[0]), float(p[1])) for p in a]
    pts_b = [(float(p[0]), float(p[1])) for p in b]
    xs = [p[0] for p in pts_a + pts_b]
    ys = [p[1] for p in pts_a + pts_b]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 1.0 or height <= 1.0:
        return 0.0

    import numpy as np

    scale = 64.0 / max(width, height)
    size = (max(2, int(math.ceil(width * scale)) + 1), max(2, int(math.ceil(height * scale)) + 1))

    def rasterize(pts: Sequence[Tuple[float, float]]) -> "np.ndarray":
        import numpy as np

        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        poly = np.array(
            [[(p[0] - min_x) * scale, (p[1] - min_y) * scale] for p in pts], dtype=np.int32
        )
        import cv2

        cv2.fillPoly(mask, [poly], 255)
        return mask

    mask_a = rasterize(pts_a)
    mask_b = rasterize(pts_b)
    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))
    return float(inter / union) if union > 0 else 0.0
