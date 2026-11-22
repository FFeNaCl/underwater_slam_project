"""扁平参数模型 + YAML 加载。

为什么是"扁平"的：ROS2 参数文件里的嵌套 map 支持有限，而数组必须是同构的
（没有"数组的数组"）。所以本项目的所有配置项都用**扁平命名**，例如
``hsv_ranges`` 用长度为 6N 的整型数组表示 N 段 HSV 区间：

    hsv_ranges: [0, 90, 60, 10, 255, 255, 170, 90, 60, 180, 255, 255]

ROS 节点与命令行工具读取**同一个 YAML 文件**：文件既可以是 ROS 参数格式

    cube_detector:
      ros__parameters:
        min_area_px: 250

也可以是纯扁平格式（没有外层节点名）。``load_flat_yaml`` 会自动识别并解包，
所以 ``calibrate_cli --config config/cube_detector.yaml`` 也能直接读 ROS 参数文件。

新增参数时必须同时改三处，否则会出现"设了没生效"的经典问题：
  1. 本文件的 dataclass 字段与 ``from_flat`` 解析；
  2. ``config/*.yaml`` 的默认值；
  3. ROS 节点里的 ``declare_parameter``（节点通过 :func:`declare_from_flat` 统一声明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

__all__ = [
    "HSV_PRESETS",
    "ColorConfig",
    "DetectConfig",
    "TrackConfig",
    "MeasureConfig",
    "CubeDetectorConfig",
    "SlamBridgeConfig",
    "CalibrationConfig",
    "LandmarkConfig",
    "load_flat_yaml",
    "dump_flat_yaml",
    "resolve_config_path",
]

#: HSV 预设，(H_low, S_low, V_low, H_high, S_high, V_high)，OpenCV 中 H∈[0,180]。
#: 红色跨越 0 度，因此需要两段；水下红橙衰减最快，饱和度/亮度阈值不能设太高。
#: 这些只是起点，现场请用 ``cube_detect_cli --print-mask`` 看分割效果后微调。
HSV_PRESETS: Dict[str, List[Tuple[int, int, int, int, int, int]]] = {
    "red": [(0, 90, 55, 10, 255, 255), (170, 90, 55, 180, 255, 255)],
    "orange": [(8, 90, 60, 22, 255, 255)],
    "yellow": [(22, 80, 70, 35, 255, 255)],
    "green": [(40, 60, 50, 85, 255, 255)],
    "cyan": [(85, 60, 50, 100, 255, 255)],
    "blue": [(100, 70, 45, 130, 255, 255)],
    "magenta": [(140, 60, 50, 169, 255, 255)],
}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return bool(default)
    return bool(value)


def _to_float_list(value: Any, length: Optional[int] = None) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        out = [float(value)]
    else:
        out = [float(v) for v in value]
    if length is not None:
        if len(out) < length:
            out = out + [0.0] * (length - len(out))
        out = out[:length]
    return out


def _to_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(v) for v in value]


def _to_str_list(value: Any) -> List[str]:
    """解析类别名列表：支持 YAML 列表或 "cube,marker" 字符串。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_flat_yaml(path: str | Path, node_name: Optional[str] = None) -> Dict[str, Any]:
    """读取 YAML 并返回扁平参数字典。

    支持的三种写法：
    1. ROS 参数文件：``<node_name>: {ros__parameters: {...}}``（``node_name`` 给定时优先取它）
    2. 带 ``ros__parameters`` 的单层 map
    3. 纯扁平 map

    Args:
        path: YAML 文件路径。
        node_name: 期望的节点名；``None`` 时自动选择唯一含 ``ros__parameters`` 的节点。

    Returns:
        扁平参数字典；文件不存在或为空时返回 ``{}``。
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}

    if node_name and node_name in data and isinstance(data[node_name], dict):
        params = data[node_name].get("ros__parameters", data[node_name])
        if isinstance(params, dict):
            return {k: v for k, v in params.items() if not str(k).startswith("#")}

    if "ros__parameters" in data and isinstance(data["ros__parameters"], dict):
        return {k: v for k, v in data["ros__parameters"].items() if not str(k).startswith("#")}

    # 多节点文件：把每个节点的 ros__parameters 合并（后面的覆盖前面的）
    merged: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "ros__parameters" in value and isinstance(value["ros__parameters"], dict):
            merged.update(value["ros__parameters"])
    if merged:
        return merged
    return {k: v for k, v in data.items() if not str(k).startswith("#")}


def dump_flat_yaml(params: Dict[str, Any], node_name: str) -> str:
    """把扁平参数写成 ROS 参数文件文本（自测与 ``--dump-config`` 用）。"""
    body = yaml.safe_dump(params, allow_unicode=True, sort_keys=False, default_flow_style=False)
    indented = "".join("    " + line + "\n" for line in body.rstrip("\n").split("\n"))
    return f"{node_name}:\n  ros__parameters:\n{indented}"


def resolve_config_path(explicit: str | Path | None, default: str | Path | None = None) -> Optional[str]:
    """把配置路径解析成存在的绝对路径；都找不到时返回 ``None``（调用方用默认值）。

    解析顺序：显式路径 -> 默认路径 -> ``None``。相对路径按当前工作目录解析。
    """
    for candidate in (explicit, default):
        if not candidate:
            continue
        p = Path(str(candidate)).expanduser()
        if p.is_file():
            return str(p.resolve())
    return None


@dataclass
class ColorConfig:
    """颜色分割参数（几何旁路的第一个证据来源：红色立方体）。"""

    preset: str = "red"
    #: 长度 6N 的整型数组：6 个一组 (H_low, S_low, V_low, H_high, S_high, V_high)
    hsv_ranges: List[int] = field(default_factory=list)
    #: 在 HSV 分割之外，再要求 R 通道显著大于 G/B（抑制水下偏色造成的橙色误检）
    require_red_dominance: bool = True
    red_dominance_ratio: float = 1.15
    #: mask 形态学清理
    morph_open_kernel: int = 3   # 去除椒盐噪点，0/1 表示关闭
    morph_close_kernel: int = 7  # 填补内部空洞，0/1 表示关闭
    #: 进入 HSV 前的 BGR 高斯模糊核（奇数，1 表示关闭）
    blur_kernel: int = 3
    #: 额外全局门限（0 表示不启用；HSV 区间里已含 S/V 上下限，这里是二次保险）
    min_saturation: int = 0
    min_value: int = 0

    def resolved_ranges(self) -> List[Tuple[int, int, int, int, int, int]]:
        """返回最终使用的 HSV 区间：显式 ``hsv_ranges`` 优先，否则用预设。"""
        if self.hsv_ranges:
            flat = _to_int_list(self.hsv_ranges)
            ranges: List[Tuple[int, int, int, int, int, int]] = []
            for i in range(0, len(flat) - 5, 6):
                ranges.append(tuple(flat[i : i + 6]))  # type: ignore[arg-type]
            if ranges:
                return ranges
        if self.preset in HSV_PRESETS:
            return list(HSV_PRESETS[self.preset])
        return list(HSV_PRESETS["red"])

    def resolved_ranges_flat(self) -> List[int]:
        """把最终 HSV 区间摊平成整型数组（ROS 参数只能是同构数组）。"""
        flat: List[int] = []
        for rng in self.resolved_ranges():
            flat.extend(int(v) for v in rng)
        return flat

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "ColorConfig":
        return ColorConfig(
            preset=str(flat.get("color_preset", "red")),
            hsv_ranges=_to_int_list(flat.get("hsv_ranges")),
            require_red_dominance=_to_bool(flat.get("require_red_dominance"), True),
            red_dominance_ratio=_to_float(flat.get("red_dominance_ratio"), 1.15),
            morph_open_kernel=_to_int(flat.get("morph_open_kernel"), 3),
            morph_close_kernel=_to_int(flat.get("morph_close_kernel"), 7),
            blur_kernel=_to_int(flat.get("blur_kernel"), 3),
            min_saturation=_to_int(flat.get("min_saturation"), 0),
            min_value=_to_int(flat.get("min_value"), 0),
        )


@dataclass
class DetectConfig:
    """几何检测与验证参数（决定"什么样的色块才算立方体"）。"""

    #: 检测前把图像缩放到该宽度以提速（0 = 不缩放）。结果坐标会还原到原图尺度。
    detect_width: int = 0
    #: 感兴趣区域 ``[x, y, w, h]`` 像素；空表示全图。水下常见的底部反光带可以裁掉。
    roi: List[int] = field(default_factory=list)
    min_area_px: int = 250
    #: 面积上限（占全图比例），用于排除大面积水面反光
    max_area_fraction: float = 0.25
    #: 多边形逼近系数（乘以周长），越小越贴合真实轮廓
    approx_epsilon_ratio: float = 0.02
    #: 顶点数范围：立方体 1 面可见=4，2 面=5，3 面=6
    min_vertices: int = 4
    max_vertices: int = 6
    #: 轮廓面积 / 最小外接矩形面积：正方形=1.0，正六边形≈0.87，细长/残缺形状更低
    min_fill_ratio: float = 0.62
    #: 轮廓面积 / 凸包面积
    min_convexity: float = 0.88
    #: 短边/长边 下限：正视面接近 1；斜视时长宽比会下降
    min_aspect_ratio: float = 0.30
    #: any | square_face（只认 4 顶点正视面）| cube_silhouette（要求内部棱线证据）
    mode: str = "any"
    #: 是否做内部棱线分析（立方体三面视角内部有 Y 型顶点 + 3 个方向的棱）
    use_interior_edge_check: bool = True
    canny_low: int = 40
    canny_high: int = 120
    #: HoughLinesP 最小线长（相对于 sqrt(轮廓面积) 的比例）
    min_edge_length_frac: float = 0.18
    #: 直线角度聚类容差（度）
    angle_cluster_tolerance_deg: float = 12.0
    #: 内部直线方向数下限（三面视角=3；square_face 模式下不检查）
    min_edge_directions: int = 2
    #: 内部直线总长 / 轮廓周长 下限
    min_edge_support: float = 0.10
    #: 归一化内部角点响应下限（0~1，Harris 响应按最大值归一）
    min_internal_corner_response: float = 0.05
    #: 综合置信度下限
    min_confidence: float = 0.45
    #: 重叠候选框的 NMS 阈值
    nms_iou: float = 0.35
    max_detections: int = 8
    #: 评分权重（会自动归一化，不必和为 1）
    weight_fill: float = 1.0
    weight_convexity: float = 0.8
    weight_aspect: float = 0.6
    weight_edge: float = 1.1
    weight_color: float = 0.9
    weight_area: float = 0.3
    #: 期望的最小/最大目标像素边长，用于面积项评分（0 表示不参与）
    preferred_side_px: float = 60.0

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "DetectConfig":
        roi_raw = flat.get("roi")
        roi = _to_int_list(roi_raw) if roi_raw else []
        return DetectConfig(
            detect_width=_to_int(flat.get("detect_width"), 0),
            roi=roi,
            min_area_px=_to_int(flat.get("min_area_px"), 250),
            max_area_fraction=_to_float(flat.get("max_area_fraction"), 0.25),
            approx_epsilon_ratio=_to_float(flat.get("approx_epsilon_ratio"), 0.02),
            min_vertices=_to_int(flat.get("min_vertices"), 4),
            max_vertices=_to_int(flat.get("max_vertices"), 6),
            min_fill_ratio=_to_float(flat.get("min_fill_ratio"), 0.62),
            min_convexity=_to_float(flat.get("min_convexity"), 0.88),
            min_aspect_ratio=_to_float(flat.get("min_aspect_ratio"), 0.30),
            mode=str(flat.get("mode", "any")),
            use_interior_edge_check=_to_bool(flat.get("use_interior_edge_check"), True),
            canny_low=_to_int(flat.get("canny_low"), 40),
            canny_high=_to_int(flat.get("canny_high"), 120),
            min_edge_length_frac=_to_float(flat.get("min_edge_length_frac"), 0.18),
            angle_cluster_tolerance_deg=_to_float(flat.get("angle_cluster_tolerance_deg"), 12.0),
            min_edge_directions=_to_int(flat.get("min_edge_directions"), 2),
            min_edge_support=_to_float(flat.get("min_edge_support"), 0.10),
            min_internal_corner_response=_to_float(flat.get("min_internal_corner_response"), 0.05),
            min_confidence=_to_float(flat.get("min_confidence"), 0.45),
            nms_iou=_to_float(flat.get("nms_iou"), 0.35),
            max_detections=_to_int(flat.get("max_detections"), 8),
            weight_fill=_to_float(flat.get("weight_fill"), 1.0),
            weight_convexity=_to_float(flat.get("weight_convexity"), 0.8),
            weight_aspect=_to_float(flat.get("weight_aspect"), 0.6),
            weight_edge=_to_float(flat.get("weight_edge"), 1.1),
            weight_color=_to_float(flat.get("weight_color"), 0.9),
            weight_area=_to_float(flat.get("weight_area"), 0.3),
            preferred_side_px=_to_float(flat.get("preferred_side_px"), 60.0),
        )


@dataclass
class TrackConfig:
    """跨帧跟踪参数（给语义地标提供稳定 ID；也让输出不闪烁）。"""

    enabled: bool = True
    iou_threshold: float = 0.25
    #: 连续命中多少帧才对外发布（抑制单帧误检）
    min_hits: int = 2
    #: 连续丢失多少帧后删除轨迹
    max_age: int = 8
    #: 中心点单帧最大跳变（像素），0 表示不限制
    max_center_jump_px: float = 0.0
    #: 未确认轨迹是否也发布（调试时开启，演示时关闭）
    publish_unconfirmed: bool = False

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "TrackConfig":
        return TrackConfig(
            enabled=_to_bool(flat.get("enable_tracking"), True),
            iou_threshold=_to_float(flat.get("track_iou_threshold"), 0.25),
            min_hits=_to_int(flat.get("track_min_hits"), 2),
            max_age=_to_int(flat.get("track_max_age"), 8),
            max_center_jump_px=_to_float(flat.get("track_max_center_jump_px"), 0.0),
            publish_unconfirmed=_to_bool(flat.get("track_publish_unconfirmed"), False),
        )


@dataclass
class MeasureConfig:
    """单目测量参数。已知立方体边长即可恢复尺度，这是单目最容易做对的一步。"""

    enabled: bool = True
    #: 立方体真实边长（米）。测量默认值 0.05 m = 5 cm，请按实物修改。
    cube_edge_m: float = 0.05
    #: 相机内参（像素）。为空时无法输出 bearing / distance，只能输出像素量。
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    #: 畸变系数（k1 k2 p1 p2 k3），为空按 0 处理
    dist_coeffs: List[float] = field(default_factory=list)
    #: 目标像素边长小于该值时不再给距离（误差过大）
    min_side_px_for_distance: float = 12.0
    #: 是否启用"轮廓拟合"求 6DoF 位姿与距离。开启后斜视距离误差从 ~25% 降到 <1%，
    #: 代价是每帧多花 1~30 ms（正视 1 ms 早退，斜视约 30 ms）。追求帧率可关闭。
    use_silhouette_fit: bool = True
    #: 轮廓拟合残差门限（像素）：超过则认为拟合不可信，退回"已知边长"近似。
    #: 实测合成数据上正常拟合的残差为 0.3~4.5 px（近距目标因轮廓量化偏大）。
    fit_max_rms_px: float = 5.0

    def camera_matrix(self):
        """返回 ``(3, 3)`` numpy 内参矩阵；参数不全时返回 ``None``。"""
        if None in (self.fx, self.fy, self.cx, self.cy):
            return None
        import numpy as np

        return np.array(
            [[float(self.fx), 0.0, float(self.cx)], [0.0, float(self.fy), float(self.cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def distortion(self):
        """返回 ``(1, N)`` numpy 畸变系数；未配置时返回全 0 的 5 维。"""
        import numpy as np

        coeffs = _to_float_list(self.dist_coeffs, 5)
        return np.array([coeffs], dtype=np.float64)

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "MeasureConfig":
        def _opt(key: str) -> Optional[float]:
            value = flat.get(key)
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return MeasureConfig(
            enabled=_to_bool(flat.get("enable_measurement"), True),
            cube_edge_m=_to_float(flat.get("cube_edge_m"), 0.05),
            fx=_opt("camera_fx"),
            fy=_opt("camera_fy"),
            cx=_opt("camera_cx"),
            cy=_opt("camera_cy"),
            dist_coeffs=_to_float_list(flat.get("dist_coeffs")),
            min_side_px_for_distance=_to_float(flat.get("min_side_px_for_distance"), 12.0),
            use_silhouette_fit=_to_bool(flat.get("use_silhouette_fit"), True),
            fit_max_rms_px=_to_float(flat.get("fit_max_rms_px"), 5.0),
        )


@dataclass
class CubeDetectorConfig:
    """几何旁路立方体检测器的完整配置（CLI 与 ROS 节点共用）。"""

    color: ColorConfig = field(default_factory=ColorConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    class_id: int = 0
    class_name: str = "cube"

    # ROS 相关（core 不使用，仅节点读取）
    input_topic: str = "/image_enhanced"
    annotated_topic: str = "/detections/annotated"
    detections_topic: str = "/detections"
    markers_topic: str = "/detections/markers"
    publish_annotated: bool = True
    publish_markers: bool = True
    #: 处理限频（Hz），0 = 不限频。几何法很快，通常不需要限制。
    max_rate_hz: float = 0.0

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "CubeDetectorConfig":
        return CubeDetectorConfig(
            color=ColorConfig.from_flat(flat),
            detect=DetectConfig.from_flat(flat),
            track=TrackConfig.from_flat(flat),
            measure=MeasureConfig.from_flat(flat),
            class_id=_to_int(flat.get("class_id"), 0),
            class_name=str(flat.get("class_name", "cube")),
            input_topic=str(flat.get("input_topic", "/image_enhanced")),
            annotated_topic=str(flat.get("annotated_topic", "/detections/annotated")),
            detections_topic=str(flat.get("detections_topic", "/detections")),
            markers_topic=str(flat.get("markers_topic", "/detections/markers")),
            publish_annotated=_to_bool(flat.get("publish_annotated"), True),
            publish_markers=_to_bool(flat.get("publish_markers"), True),
            max_rate_hz=_to_float(flat.get("max_rate_hz"), 0.0),
        )

    def to_flat(self) -> Dict[str, Any]:
        """展开成扁平 dict，**键名与** :meth:`from_flat` **完全一致**。

        这一点是硬约束：任何"节名前缀"式的自造键（例如 ``color_preset`` 之外的
        ``color_hsv_ranges``）都会导致 round-trip 丢参数。单元测试
        ``test_params_roundtrip`` 会校验 ``from_flat(to_flat()) == 自身``。
        """
        out: Dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "input_topic": self.input_topic,
            "annotated_topic": self.annotated_topic,
            "detections_topic": self.detections_topic,
            "markers_topic": self.markers_topic,
            "publish_annotated": self.publish_annotated,
            "publish_markers": self.publish_markers,
            "max_rate_hz": self.max_rate_hz,
            # --- ColorConfig ---
            "color_preset": self.color.preset,
            "hsv_ranges": self.color.resolved_ranges_flat(),
            "require_red_dominance": self.color.require_red_dominance,
            "red_dominance_ratio": self.color.red_dominance_ratio,
            "morph_open_kernel": self.color.morph_open_kernel,
            "morph_close_kernel": self.color.morph_close_kernel,
            "blur_kernel": self.color.blur_kernel,
            "min_saturation": self.color.min_saturation,
            "min_value": self.color.min_value,
            # --- DetectConfig ---
            "detect_width": self.detect.detect_width,
            "roi": list(self.detect.roi),
            "min_area_px": self.detect.min_area_px,
            "max_area_fraction": self.detect.max_area_fraction,
            "approx_epsilon_ratio": self.detect.approx_epsilon_ratio,
            "min_vertices": self.detect.min_vertices,
            "max_vertices": self.detect.max_vertices,
            "min_fill_ratio": self.detect.min_fill_ratio,
            "min_convexity": self.detect.min_convexity,
            "min_aspect_ratio": self.detect.min_aspect_ratio,
            "mode": self.detect.mode,
            "use_interior_edge_check": self.detect.use_interior_edge_check,
            "canny_low": self.detect.canny_low,
            "canny_high": self.detect.canny_high,
            "min_edge_length_frac": self.detect.min_edge_length_frac,
            "angle_cluster_tolerance_deg": self.detect.angle_cluster_tolerance_deg,
            "min_edge_directions": self.detect.min_edge_directions,
            "min_edge_support": self.detect.min_edge_support,
            "min_internal_corner_response": self.detect.min_internal_corner_response,
            "min_confidence": self.detect.min_confidence,
            "nms_iou": self.detect.nms_iou,
            "max_detections": self.detect.max_detections,
            "weight_fill": self.detect.weight_fill,
            "weight_convexity": self.detect.weight_convexity,
            "weight_aspect": self.detect.weight_aspect,
            "weight_edge": self.detect.weight_edge,
            "weight_color": self.detect.weight_color,
            "weight_area": self.detect.weight_area,
            "preferred_side_px": self.detect.preferred_side_px,
            # --- TrackConfig ---
            "enable_tracking": self.track.enabled,
            "track_iou_threshold": self.track.iou_threshold,
            "track_min_hits": self.track.min_hits,
            "track_max_age": self.track.max_age,
            "track_max_center_jump_px": self.track.max_center_jump_px,
            "track_publish_unconfirmed": self.track.publish_unconfirmed,
            # --- MeasureConfig ---
            "enable_measurement": self.measure.enabled,
            "cube_edge_m": self.measure.cube_edge_m,
            "camera_fx": self.measure.fx,
            "camera_fy": self.measure.fy,
            "camera_cx": self.measure.cx,
            "camera_cy": self.measure.cy,
            "dist_coeffs": list(self.measure.dist_coeffs),
            "min_side_px_for_distance": self.measure.min_side_px_for_distance,
            "use_silhouette_fit": self.measure.use_silhouette_fit,
            "fit_max_rms_px": self.measure.fit_max_rms_px,
        }
        return out


@dataclass
class SlamBridgeConfig:
    """图像 -> ORB-SLAM3 C++ 节点 的桥接参数。"""

    image_topic: str = "/image_enhanced"
    #: ORB-SLAM3 封装包（ros2_orb_slam3）硬编码的输入话题，不要改，除非同步改 C++
    target_image_topic: str = "/mono_py_driver/img_msg"
    target_timestep_topic: str = "/mono_py_driver/timestep_msg"
    config_request_topic: str = "/mono_py_driver/experiment_settings"
    config_ack_topic: str = "/mono_py_driver/exp_settings_ack"
    #: ORB-SLAM3 设置文件名（不含 .yaml），对应 <settings_dir>/<settings_name>.yaml
    settings_name: str = "water_1280x480"
    #: 是否用图像头时间戳；False 时用单调时钟（数据集/仿真常用）
    use_header_stamp: bool = True
    #: 发布限频（Hz），0 = 跟随输入
    max_rate_hz: float = 0.0
    #: 与上一帧的最小时间间隔（秒），防止重复时间戳导致 SLAM 丢帧
    min_dt_s: float = 0.001
    #: 保持向 C++ 节点重发配置直到收到 ACK 的间隔（秒）
    handshake_retry_s: float = 0.5
    #: 收到 ACK 后是否继续周期性重发（C++ 节点重启后能自动恢复）
    rehandshake_period_s: float = 0.0

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "SlamBridgeConfig":
        return SlamBridgeConfig(
            image_topic=str(flat.get("image_topic", "/image_enhanced")),
            target_image_topic=str(flat.get("target_image_topic", "/mono_py_driver/img_msg")),
            target_timestep_topic=str(flat.get("target_timestep_topic", "/mono_py_driver/timestep_msg")),
            config_request_topic=str(flat.get("config_request_topic", "/mono_py_driver/experiment_settings")),
            config_ack_topic=str(flat.get("config_ack_topic", "/mono_py_driver/exp_settings_ack")),
            settings_name=str(flat.get("settings_name", "water_1280x480")),
            use_header_stamp=_to_bool(flat.get("use_header_stamp"), True),
            max_rate_hz=_to_float(flat.get("max_rate_hz"), 0.0),
            min_dt_s=_to_float(flat.get("min_dt_s"), 0.001),
            handshake_retry_s=_to_float(flat.get("handshake_retry_s"), 0.5),
            rehandshake_period_s=_to_float(flat.get("rehandshake_period_s"), 0.0),
        )


@dataclass
class CalibrationConfig:
    """标定参数（棋盘格 8x6 / 25mm 为当前实物的默认值）。"""

    #: chessboard | circles | charuco
    pattern: str = "chessboard"
    #: 内角点列数 x 行数（8x6 棋盘格 = 7x5 方格）
    cols: int = 8
    rows: int = 6
    #: 方格真实边长（米）
    square_size: float = 0.025
    #: 采集要求
    min_views: int = 12
    max_views: int = 40
    #: 同一视图内必须检出的最少角点数（过滤半可见棋盘）
    min_corners_ratio: float = 0.85
    #: 角点多样性：不同视图中心的网格划分数量（1 = 不检查）
    coverage_grid: int = 3
    min_coverage_cells: int = 3
    #: 亚像素优化窗口
    subpix_window: int = 11
    #: 标定选项
    fix_k3: bool = False
    fix_aspect_ratio: bool = False
    zero_tangent_dist: bool = False
    #: 去畸变输出图保留的有效像素比例（0=全保留有黑边，1=裁掉黑边）
    undistort_alpha: float = 0.0
    #: 输出
    camera_name: str = "water_camera"
    output_dir: str = "calibration_out"
    orbslam3_settings_name: str = "water_1280x480"
    fps: float = 30.0
    save_undistorted: bool = True
    save_debug_corners: bool = True

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """返回 ``(列, 行)`` 内角点数。"""
        return (int(self.cols), int(self.rows))

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "CalibrationConfig":
        return CalibrationConfig(
            pattern=str(flat.get("pattern", "chessboard")),
            cols=_to_int(flat.get("cols"), 8),
            rows=_to_int(flat.get("rows"), 6),
            square_size=_to_float(flat.get("square_size"), 0.025),
            min_views=_to_int(flat.get("min_views"), 12),
            max_views=_to_int(flat.get("max_views"), 40),
            min_corners_ratio=_to_float(flat.get("min_corners_ratio"), 0.85),
            coverage_grid=_to_int(flat.get("coverage_grid"), 3),
            min_coverage_cells=_to_int(flat.get("min_coverage_cells"), 3),
            subpix_window=_to_int(flat.get("subpix_window"), 11),
            fix_k3=_to_bool(flat.get("fix_k3"), False),
            fix_aspect_ratio=_to_bool(flat.get("fix_aspect_ratio"), False),
            zero_tangent_dist=_to_bool(flat.get("zero_tangent_dist"), False),
            undistort_alpha=_to_float(flat.get("undistort_alpha"), 0.0),
            camera_name=str(flat.get("camera_name", "water_camera")),
            output_dir=str(flat.get("output_dir", "calibration_out")),
            orbslam3_settings_name=str(flat.get("orbslam3_settings_name", "water_1280x480")),
            fps=_to_float(flat.get("fps"), 30.0),
            save_undistorted=_to_bool(flat.get("save_undistorted"), True),
            save_debug_corners=_to_bool(flat.get("save_debug_corners"), True),
        )


@dataclass
class LandmarkConfig:
    """语义地标（多视图三角化）参数。"""

    detections_topic: str = "/detections"
    pose_topic: str = "/slam/pose"
    status_topic: str = "/slam/status"
    landmarks_topic: str = "/semantic/landmarks"
    markers_topic: str = "/semantic/markers"
    #: 需要多少不同视角才允许三角化
    min_views: int = 2
    #: 两条射线的最小夹角（度）：太小则三角化病态
    min_parallax_deg: float = 2.0
    #: 两条射线的最小基线（米）：单目 SLAM 尺度由初始化决定，这里只做几何过滤
    min_baseline_m: float = 0.02
    #: 三角化重投影误差上限（像素），超过则丢弃该观测
    max_reproj_error_px: float = 12.0
    #: 同一 track_id 的观测最多保留多少个（滑动窗口）
    max_observations: int = 30
    #: 超过该时间未更新则停止发布（秒）
    stale_after_s: float = 5.0
    #: 是否在没有 SLAM 位姿时用"已知边长 + 像素方位"给出相机系下的近似点
    allow_scale_fallback: bool = True
    #: 立方体边长（米），scale fallback 与 marker 尺寸都用它
    cube_edge_m: float = 0.05
    #: 只允许这些类别成为地标（空 = 不过滤）。真实 YOLO 多类别时建议填静态目标，
    #: 例如 ["cube", "marker"]，避免鱼/气泡等动态目标进地图。
    landmark_class_names: List[str] = field(default_factory=list)
    #: 检测置信度下限（低于该值不参与三角化）
    min_detection_confidence: float = 0.0

    def accepts(self, class_name: str, confidence: float) -> bool:
        """判断一次检测是否允许进入地标管理器。"""
        if float(confidence) < float(self.min_detection_confidence):
            return False
        if self.landmark_class_names and str(class_name) not in self.landmark_class_names:
            return False
        return True

    @staticmethod
    def from_flat(flat: Dict[str, Any]) -> "LandmarkConfig":
        return LandmarkConfig(
            detections_topic=str(flat.get("detections_topic", "/detections")),
            pose_topic=str(flat.get("pose_topic", "/slam/pose")),
            status_topic=str(flat.get("status_topic", "/slam/status")),
            landmarks_topic=str(flat.get("landmarks_topic", "/semantic/landmarks")),
            markers_topic=str(flat.get("markers_topic", "/semantic/markers")),
            min_views=_to_int(flat.get("min_views"), 2),
            min_parallax_deg=_to_float(flat.get("min_parallax_deg"), 2.0),
            min_baseline_m=_to_float(flat.get("min_baseline_m"), 0.02),
            max_reproj_error_px=_to_float(flat.get("max_reproj_error_px"), 12.0),
            max_observations=_to_int(flat.get("max_observations"), 30),
            stale_after_s=_to_float(flat.get("stale_after_s"), 5.0),
            allow_scale_fallback=_to_bool(flat.get("allow_scale_fallback"), True),
            cube_edge_m=_to_float(flat.get("cube_edge_m"), 0.05),
            landmark_class_names=_to_str_list(flat.get("landmark_class_names")),
            min_detection_confidence=_to_float(flat.get("min_detection_confidence"), 0.0),
        )


def declare_from_flat(node, flat: Dict[str, Any], skip: Iterable[str] = ()) -> None:
    """把扁平参数声明到 ROS 节点上（供 nodes/ 使用）。

    Args:
        node: ``rclpy.node.Node`` 实例。
        flat: 参数名 -> 默认值。
        skip: 不声明的键（例如已经由节点单独声明的 config_file）。

    Notes:
        ROS 参数的合法类型只有 bool / int / double / string 及其同构数组，
        这里按 Python 值类型映射；``None`` 会被跳过（用节点自己的默认值）。
    """
    skip_set = set(skip)
    for key, value in flat.items():
        if key in skip_set or value is None:
            continue
        if isinstance(value, bool):
            node.declare_parameter(key, value)
        elif isinstance(value, int):
            node.declare_parameter(key, value)
        elif isinstance(value, float):
            node.declare_parameter(key, float(value))
        elif isinstance(value, str):
            node.declare_parameter(key, value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                node.declare_parameter(key, [])
            elif all(isinstance(v, bool) for v in value):
                node.declare_parameter(key, [bool(v) for v in value])
            elif all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                node.declare_parameter(key, [int(v) for v in value])
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                node.declare_parameter(key, [float(v) for v in value])
            else:
                node.declare_parameter(key, [str(v) for v in value])
        elif isinstance(value, dict):
            continue  # 扁平化设计：不应出现


def flat_from_node(node, source: Dict[str, Any]) -> Dict[str, Any]:
    """从 ROS 节点读取已声明的参数，返回扁平 dict（用于构造 core 配置）。"""
    out: Dict[str, Any] = {}
    for key, value in source.items():
        if value is None:
            continue
        try:
            out[key] = node.get_parameter(key).value
        except Exception:  # pragma: no cover - 未声明的参数
            continue
    return out
