"""相机标定：角点检测、标定求解、质量评估与输出文件生成。

**单位与约定**
* ``square_size`` 用米，标定得到的 ``tvec`` 也以米为单位；内参 ``fx fy cx cy`` 是像素。
* 棋盘格尺寸参数一律指**内角点数**，与 ROS ``camera_calibration --size 8x6`` 一致。
  8x6 内角点 = 9x7 个方格。当前实物默认：8x6 内角点、方格 25 mm。
* 物体坐标系：``x`` 向右、``y`` 向下、``z`` 垂直于板面（详见 core/synthetic.py）。

**输出四件套**（:func:`write_all_outputs` 一次写全）::

    ost.yaml                       ROS camera_calibration 兼容格式（可直接被别的节点读）
    water_1280x480.yaml            ORB-SLAM3 设置文件（放到 orb_slam3/config/Monocular/）
    enhancer_camera_params.yaml    图像增强节点用的 camera_matrix / distortion_coeffs
    calibration_report.json        全部量化指标与警告（AI/报告用）

**水下标定的现实提醒**（写进报告，答辩时也建议说明）
平面防水壳会引入"空气-玻璃-水"多介质折射，普通针孔+径向畸变模型只是近似：
* 越靠画面边缘误差越大，因此**不要**用极端掠射角/极限边缘的视图去刷低 RMS；
* 标定与使用时若"防水壳、水深、镜头到玻璃的距离"改变，内参需重标；
* 若追求更高精度，应换球面端口或引入折射模型（本项目暂不实现）。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .cv_compat import find_chessboard_corners, to_gray
from .params import CalibrationConfig
from .synthetic import chessboard_object_points

__all__ = [
    "ViewResult",
    "CalibrationResult",
    "detect_pattern",
    "view_from_array",
    "view_from_path",
    "collect_views",
    "calibrate",
    "undistort_image",
    "write_all_outputs",
    "write_ost_yaml",
    "write_orbslam3_yaml",
    "write_enhancer_yaml",
    "write_report",
]


@dataclass
class ViewResult:
    """单个视图的角点检测结果。"""

    corners: Optional[np.ndarray]          # (N, 1, 2) float32
    object_points: Optional[np.ndarray]    # (N, 3) float64
    image_size: Tuple[int, int]            # (宽, 高)
    source: str = ""                       # 文件路径或 "<array>"
    method: str = "NONE"                   # SB / CLASSIC / NONE
    corner_ratio: float = 0.0              # 检出角点数 / 期望角点数
    error_px: Optional[float] = None       # 标定后的重投影误差（像素）
    coverage_cells: int = 0                # 覆盖到的网格数（多样性）
    accepted: bool = False
    reason: str = ""                       # 未接受的原因

    @property
    def found(self) -> bool:
        return self.corners is not None and len(self.corners) > 0


@dataclass
class CalibrationResult:
    """标定结果（含质量指标与警告）。"""

    ok: bool
    camera_matrix: Optional[np.ndarray]
    dist_coeffs: Optional[np.ndarray]
    image_size: Tuple[int, int]
    rms: float = 0.0
    n_views_used: int = 0
    n_views_total: int = 0
    per_view_errors: List[float] = field(default_factory=list)
    rvecs: List[np.ndarray] = field(default_factory=list)
    tvecs: List[np.ndarray] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config: Optional[CalibrationConfig] = None
    source: str = "unknown"
    message: str = ""

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0]) if self.camera_matrix is not None else 0.0

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1]) if self.camera_matrix is not None else 0.0

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2]) if self.camera_matrix is not None else 0.0

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2]) if self.camera_matrix is not None else 0.0

    def distortion_list(self, length: int = 5) -> List[float]:
        """返回长度固定的畸变系数列表（不足补 0）。"""
        if self.dist_coeffs is None:
            return [0.0] * length
        flat = [float(v) for v in np.asarray(self.dist_coeffs).reshape(-1)]
        if len(flat) < length:
            flat += [0.0] * (length - len(flat))
        return flat[:length]

    def quality_grade(self) -> str:
        """按 RMS 与视图数给出粗略等级（仅用于自检提示，不替代人工判断）。"""
        if not self.ok:
            return "failed"
        if self.rms <= 0.3 and self.n_views_used >= 15:
            return "good"
        if self.rms <= 0.8 and self.n_views_used >= 10:
            return "fair"
        return "poor"

    def report_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的指标字典。"""
        return {
            "ok": self.ok,
            "message": self.message,
            "image_size": {"width": int(self.image_size[0]), "height": int(self.image_size[1])},
            "rms_reprojection_px": float(self.rms),
            "views_used": int(self.n_views_used),
            "views_total": int(self.n_views_total),
            "per_view_error_px": [float(v) for v in self.per_view_errors],
            "camera_matrix": None if self.camera_matrix is None else [[float(v) for v in row] for row in self.camera_matrix],
            "distortion_coeffs": self.distortion_list(5),
            "quality_grade": self.quality_grade(),
            "warnings": list(self.warnings),
            "pinhole_approx_note": (
                "平面防水壳存在空气-玻璃-水折射，针孔+径向畸变仅为近似；"
                "壳/水深/镜头-玻璃距离改变后需要重标。"
            ),
            "source": self.source,
        }


def validate_corners(
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    max_rms_px: float = 3.0,
) -> Tuple[bool, float]:
    """用"角点栅格应当是点阵的透视投影"来校验检出结果（防止检测器锁到错网格）。

    做法：以理想的规整点阵与检出的角点建立单应矩阵，若残差很小，说明这 48 个点
    确实构成一个规整的棋盘点阵；残差很大则说明检测器锁到了偏移、外扩或残缺的图案
    （开发过程中真实遇到过：合成图缺少静区时，检出的点阵整体偏移一格并延伸出板外，
    单独看"检出 48 个角点"完全正常，只有几何校验能发现）。

    这个检查在真实数据上同样有用：光照不均、板子部分被遮挡时，
    ``findChessboardCorners`` 仍可能返回"看起来很成功"的错误点阵。

    Args:
        corners: 检出角点，形状 ``(N, 1, 2)`` 或 ``(N, 2)``。
        pattern_size: ``(列, 行)`` 内角点数。
        max_rms_px: 允许的单应残差上限（像素）。

    Returns:
        ``(valid, rms_px)``。
    """
    if corners is None:
        return False, float("inf")
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    if len(points) != cols * rows:
        return False, float("inf")

    ideal = np.zeros((cols * rows, 2), dtype=np.float64)
    ideal[:, 0] = np.tile(np.arange(cols), rows)
    ideal[:, 1] = np.repeat(np.arange(rows), cols)

    try:
        homography, _ = cv2.findHomography(ideal, points)
    except cv2.error:  # pragma: no cover - 退化输入
        return False, float("inf")
    if homography is None:
        return False, float("inf")

    projected = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = projected - points
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return bool(rms <= float(max_rms_px)), rms


def detect_pattern(
    image: np.ndarray,
    config: CalibrationConfig,
    subpix_window: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """在单张图上检测标定板内角点（当前主路径：棋盘格）。

    Args:
        image: BGR 或灰度图像。
        config: 标定配置（图案类型、内角点数、方格边长）。
        subpix_window: 亚像素优化窗口，``None`` 时用配置值。

    Returns:
        ``(corners, method)``；未检出或未通过几何校验时 ``(None, "NONE")``。
    """
    gray = to_gray(image)
    pattern_size = config.pattern_size
    pattern = (config.pattern or "chessboard").lower()

    if pattern in ("chessboard", "checkerboard"):
        found, corners, method = find_chessboard_corners(gray, pattern_size, use_sb=True)
        if not found or corners is None:
            return None, "NONE"
        valid, rms = validate_corners(corners, pattern_size)
        if not valid:
            return None, f"NONE(grid_check_failed:{rms:.1f}px)"
        window = int(subpix_window or config.subpix_window)
        if window > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (window, window), (-1, -1), criteria)
        return corners, method

    if pattern in ("circles", "acircles", "circles_grid"):
        flags = cv2.CALIB_CB_SYMMETRIC_GRID
        found, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)
        if found and corners is not None:
            return corners, "CIRCLES"
        return None, "NONE"

    # ChArUco 需要 opencv-contrib，且 4.7 前后 API 不同：尽力支持，失败即返回 NONE
    if pattern == "charuco":
        try:  # pragma: no cover - 依赖可选 contrib
            aruco = cv2.aruco
            dictionary = aruco.getPredefinedDictionary(getattr(aruco, "DICT_5X5_100", 0))
            if hasattr(aruco, "CharucoDetector"):
                board = aruco.CharucoBoard(pattern_size, config.square_size, config.square_size * 0.5, dictionary)
                detector = aruco.CharucoDetector(board)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
                if charuco_corners is not None and len(charuco_corners) > 0:
                    return np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2), "CHARUCO"
            else:
                board = aruco.CharucoBoard_create(
                    pattern_size[0], pattern_size[1], config.square_size, config.square_size * 0.5, dictionary
                )
                _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    cv2.aruco.detectMarkers(gray, dictionary)[0], cv2.aruco.detectMarkers(gray, dictionary)[1], gray, board
                )
                if charuco_corners is not None and len(charuco_corners) > 0:
                    return np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2), "CHARUCO"
        except Exception:
            return None, "NONE"
        return None, "NONE"

    raise ValueError(f"unsupported pattern: {config.pattern}")


def _coverage_cells(corners: np.ndarray, image_size: Tuple[int, int], grid: int) -> int:
    """统计角点覆盖到多少个网格单元（衡量姿态/位置多样性）。"""
    if corners is None or len(corners) == 0 or grid <= 1:
        return 0
    width, height = image_size
    centers = np.asarray(corners).reshape(-1, 2)
    cx = np.clip((centers[:, 0] / max(width, 1) * grid).astype(int), 0, grid - 1)
    cy = np.clip((centers[:, 1] / max(height, 1) * grid).astype(int), 0, grid - 1)
    return int(len(set(zip(cx.tolist(), cy.tolist()))))


def view_from_array(
    image: np.ndarray,
    config: CalibrationConfig,
    source: str = "<array>",
) -> ViewResult:
    """从内存图像构造一个 :class:`ViewResult`（含接受/拒绝判定）。"""
    height, width = image.shape[:2]
    image_size = (int(width), int(height))
    corners, method = detect_pattern(image, config)
    expected = config.cols * config.rows
    if corners is None:
        return ViewResult(None, None, image_size, source=source, method="NONE", reason="pattern_not_found")

    ratio = float(len(corners)) / float(expected) if expected else 0.0
    cells = _coverage_cells(corners, image_size, config.coverage_grid)
    accepted = ratio >= float(config.min_corners_ratio)
    reason = "" if accepted else "insufficient_corners"
    return ViewResult(
        corners=corners,
        object_points=chessboard_object_points(config.cols, config.rows, config.square_size),
        image_size=image_size,
        source=source,
        method=method,
        corner_ratio=ratio,
        coverage_cells=cells,
        accepted=accepted,
        reason=reason,
    )


def view_from_path(path: str | Path, config: CalibrationConfig) -> ViewResult:
    """从图片文件构造一个 :class:`ViewResult`。

    中文/空格路径用 ``np.fromfile + cv2.imdecode`` 读取，避免 OpenCV 在 Windows
    上对非 ASCII 路径的经典失败。
    """
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return ViewResult(None, None, (0, 0), source=str(p), method="NONE", reason="image_read_failed")
    return view_from_array(image, config, source=str(p))


def collect_views(
    images: Iterable[np.ndarray | str | Path],
    config: CalibrationConfig,
) -> Tuple[List[ViewResult], List[ViewResult]]:
    """批量采集视图，返回 ``(accepted, rejected)``。"""
    accepted: List[ViewResult] = []
    rejected: List[ViewResult] = []
    for item in images:
        if isinstance(item, (str, Path)):
            result = view_from_path(item, config)
        else:
            result = view_from_array(item, config)
        if result.accepted and len(accepted) < int(config.max_views):
            accepted.append(result)
        elif not result.accepted:
            rejected.append(result)
    return accepted, rejected


def calibrate(
    views: Sequence[ViewResult],
    image_size: Tuple[int, int],
    config: Optional[CalibrationConfig] = None,
    source: str = "views",
) -> CalibrationResult:
    """执行标定并计算每个视图的重投影误差。

    Args:
        views: 通过筛选的视图（``ViewResult``）。
        image_size: ``(宽, 高)``，必须与实际图像一致，否则内参不可用。
        config: 标定配置（决定固定哪些参数）。
        source: 数据来源标记（``images`` / ``ros`` / ``synthetic``）。

    Returns:
        :class:`CalibrationResult`；失败时 ``ok=False`` 并在 ``message`` 里说明原因。
    """
    cfg = config or CalibrationConfig()
    warnings: List[str] = []
    total = len(views)
    usable = [v for v in views if v.found and v.object_points is not None]

    if len(usable) < 3:
        return CalibrationResult(
            ok=False,
            camera_matrix=None,
            dist_coeffs=None,
            image_size=image_size,
            n_views_used=len(usable),
            n_views_total=total,
            warnings=["视图不足：标定至少需要 3 个有效视图，推荐 15 个以上"],
            config=cfg,
            source=source,
            message="not_enough_views",
        )

    object_points = [np.asarray(v.object_points, dtype=np.float32).reshape(-1, 1, 3) for v in usable]
    image_points = [np.asarray(v.corners, dtype=np.float32).reshape(-1, 1, 2) for v in usable]
    size = (int(image_size[0]), int(image_size[1]))

    flags = 0
    if cfg.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if cfg.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if cfg.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, size, None, None, flags=flags
        )
    except cv2.error as exc:  # pragma: no cover - 由 OpenCV 抛出的求解失败
        return CalibrationResult(
            ok=False,
            camera_matrix=None,
            dist_coeffs=None,
            image_size=size,
            n_views_used=len(usable),
            n_views_total=total,
            warnings=[f"cv2.calibrateCamera failed: {exc}"],
            config=cfg,
            source=source,
            message="calibration_error",
        )

    # 逐视图重投影误差
    errors: List[float] = []
    for index, view in enumerate(usable):
        projected, _ = cv2.projectPoints(
            object_points[index], rvecs[index], tvecs[index], camera_matrix, dist_coeffs
        )
        diff = projected.reshape(-1, 2) - image_points[index].reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))
        view.error_px = errors[-1]

    if len(usable) < cfg.min_views:
        warnings.append(
            f"有效视图仅 {len(usable)} 个，少于建议值 {cfg.min_views} 个：内参与畸变系数可能不稳定"
        )
    if rms > 1.0:
        warnings.append(f"RMS 重投影误差 {rms:.3f} px 偏大（>1.0），请检查棋盘平整度、运动模糊与角点检测质量")
    if max(errors) > 3.0 * max(rms, 1e-6):
        worst_view = int(np.argmax(errors))
        warnings.append(
            f"视图 {usable[worst_view].source} 的误差 {max(errors):.2f} px 明显高于均值，建议剔除后重标"
        )
    coverage = sum(1 for v in usable if v.coverage_cells >= cfg.min_coverage_cells)
    if coverage < len(usable):
        warnings.append(
            f"有 {len(usable) - coverage} 个视图的角点分布覆盖不足（网格 {cfg.coverage_grid}x{cfg.coverage_grid}），"
            "请让棋盘出现在画面不同区域与角度"
        )
    warnings.append(
        "水下标定使用针孔+径向畸变近似；防水壳、水深、镜头到玻璃距离改变后必须重标。"
    )

    return CalibrationResult(
        ok=True,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=size,
        rms=float(rms),
        n_views_used=len(usable),
        n_views_total=total,
        per_view_errors=errors,
        rvecs=[np.asarray(r) for r in rvecs],
        tvecs=[np.asarray(t) for t in tvecs],
        warnings=warnings,
        config=cfg,
        source=source,
        message="ok",
    )


def undistort_image(
    image: np.ndarray,
    result: CalibrationResult,
    alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """去畸变，返回 ``(图像, 新内参, ROI)``。

    ``alpha=0`` 保留全部像素（有黑边，适合标定/检测）；``alpha=1`` 裁掉黑边
    （适合给人看的演示图）。ROI 为 ``(x, y, w, h)``。
    """
    if result.camera_matrix is None:
        raise ValueError("calibration result has no camera matrix")
    dist = np.asarray(result.dist_coeffs if result.dist_coeffs is not None else np.zeros((1, 5)), dtype=np.float64)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        result.camera_matrix, dist, result.image_size, float(alpha), result.image_size
    )
    undistorted = cv2.undistort(image, result.camera_matrix, dist, None, new_matrix)
    return undistorted, new_matrix, tuple(int(v) for v in roi)


def _distortion_rows(values: Sequence[float]) -> Dict[str, Any]:
    return {"rows": 1, "cols": int(len(values)), "data": [float(v) for v in values]}


def write_ost_yaml(path: str | Path, result: CalibrationResult, config: Optional[CalibrationConfig] = None) -> str:
    """写出 ROS ``camera_calibration`` 兼容的 ``ost.yaml``。"""
    cfg = config or result.config or CalibrationConfig()
    if result.camera_matrix is None:
        raise ValueError("cannot write ost.yaml without a calibration result")
    dist = result.distortion_list(5)
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        result.camera_matrix,
        np.asarray(dist, dtype=np.float64).reshape(1, -1),
        result.image_size,
        float(cfg.undistort_alpha),
        result.image_size,
    )
    payload = {
        "image_width": int(result.image_size[0]),
        "image_height": int(result.image_size[1]),
        "camera_name": str(cfg.camera_name),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in np.asarray(result.camera_matrix).reshape(-1)],
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": _distortion_rows(dist),
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                float(new_matrix[0, 0]), 0.0, float(new_matrix[0, 2]), 0.0,
                0.0, float(new_matrix[1, 1]), float(new_matrix[1, 2]), 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        },
        # camera_calibration 会写这个字段，保留以便下游脚本统一解析
        "avg_reprojection_error": float(result.rms),
    }
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    header = (
        "# 由 underwater_perception/tools/calibrate_cli.py 生成\n"
        f"# 视图数: {result.n_views_used}/{result.n_views_total}, RMS={result.rms:.4f} px, 等级={result.quality_grade()}\n"
        "# 单位: 内参为像素; 方格边长 %.4f m; 图案 %dx%d 内角点\n"
        % (float(cfg.square_size), int(cfg.cols), int(cfg.rows))
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + text, encoding="utf-8")
    return str(out)


#: ORB-SLAM3 水下低纹理场景的启动参数（与 config/water_1280x480.orbslam3.example.yaml 一致）
ORBSLAM3_ORB_DEFAULTS: Dict[str, Any] = {
    "ORBextractor.nFeatures": 1600,
    "ORBextractor.scaleFactor": 1.2,
    "ORBextractor.nLevels": 8,
    "ORBextractor.iniThFAST": 12,
    "ORBextractor.minThFAST": 5,
}

#: Pangolin 可视化参数（不影响求解，只影响窗口观感）
ORBSLAM3_VIEWER_DEFAULTS: Dict[str, float] = {
    "Viewer.KeyFrameSize": 0.05,
    "Viewer.KeyFrameLineWidth": 1.0,
    "Viewer.GraphLineWidth": 0.9,
    "Viewer.PointSize": 2.0,
    "Viewer.CameraSize": 0.08,
    "Viewer.CameraLineWidth": 3.0,
    "Viewer.ViewpointX": 0.0,
    "Viewer.ViewpointY": -0.7,
    "Viewer.ViewpointZ": -1.8,
    "Viewer.ViewpointF": 500.0,
}


def write_orbslam3_yaml(
    path: str | Path,
    result: CalibrationResult,
    config: Optional[CalibrationConfig] = None,
    fps: Optional[float] = None,
    orb_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """写出 ORB-SLAM3 单目设置文件（``%YAML:1.0`` 风格，键名必须完全一致）。

    注意 ORB-SLAM3 读取的是 OpenCV FileStorage 格式：**键名区分大小写且不能改**，
    值为数字时不能加引号（``Camera.type`` 例外，它是字符串）。
    """
    cfg = config or result.config or CalibrationConfig()
    dist = result.distortion_list(5)
    width, height = int(result.image_size[0]), int(result.image_size[1])
    orb = dict(ORBSLAM3_ORB_DEFAULTS)
    if orb_overrides:
        orb.update(orb_overrides)

    lines: List[str] = [
        "%YAML:1.0",
        "",
        "# 由 underwater_perception 标定程序生成，可直接放到 ros2_orb_slam3/orb_slam3/config/Monocular/",
        f"# 视图数 {result.n_views_used}/{result.n_views_total}，RMS={result.rms:.4f} px，等级 {result.quality_grade()}",
        "",
        'Camera.type: "PinHole"',
        f"Camera.fx: {result.fx:.6f}",
        f"Camera.fy: {result.fy:.6f}",
        f"Camera.cx: {result.cx:.6f}",
        f"Camera.cy: {result.cy:.6f}",
        "",
        f"Camera.k1: {dist[0]:.8f}",
        f"Camera.k2: {dist[1]:.8f}",
        f"Camera.p1: {dist[2]:.8f}",
        f"Camera.p2: {dist[3]:.8f}",
        f"Camera.k3: {dist[4]:.8f}",
        "",
        f"Camera.width: {width}",
        f"Camera.height: {height}",
        f"Camera.fps: {float(fps if fps is not None else cfg.fps):.1f}",
        # 本项目的图像链路统一是 BGR（image_bridge/camera 节点），所以填 0。
        # 若上游改成 RGB，请同步改成 1，否则灰度转换会交换红/蓝权重。
        "Camera.RGB: 0",
        "",
        "# 水下低纹理场景：特征点开多、FAST 阈值放低（可由 config 覆盖）",
    ]
    for key, value in orb.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.2f}")
        else:
            lines.append(f"{key}: {int(value)}")
    lines.append("")
    for key, value in ORBSLAM3_VIEWER_DEFAULTS.items():
        lines.append(f"{key}: {float(value):.2f}")
    lines.append("")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_enhancer_yaml(path: str | Path, result: CalibrationResult, config: Optional[CalibrationConfig] = None) -> str:
    """写出图像增强节点可直接使用的参数文件（``camera_matrix`` / ``distortion_coeffs``）。

    ``image_enhancer_cpp`` 用的是两个**扁平数组**参数，这里按它期望的顺序写：
    ``camera_matrix`` 为 9 个 double（行优先），``distortion_coeffs`` 为 k1 k2 p1 p2 k3。
    """
    cfg = config or result.config or CalibrationConfig()
    if result.camera_matrix is None:
        raise ValueError("cannot write enhancer params without a calibration result")
    payload = {
        "image_enhancer_cpp": {
            "ros__parameters": {
                "use_undistort": True,
                "camera_matrix": [float(v) for v in np.asarray(result.camera_matrix).reshape(-1)],
                "distortion_coeffs": result.distortion_list(5),
                "calibration_source": str(cfg.camera_name),
                "calibration_rms_px": float(result.rms),
            }
        }
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(out)


def write_report(
    path: str | Path,
    result: CalibrationResult,
    config: Optional[CalibrationConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
    views: Optional[Sequence[ViewResult]] = None,
) -> str:
    """写出 JSON 报告：指标 + 每个视图的检出情况 + 环境信息。"""
    cfg = config or result.config or CalibrationConfig()
    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "underwater_perception.core.calibration",
        "config": {
            "pattern": cfg.pattern,
            "cols": int(cfg.cols),
            "rows": int(cfg.rows),
            "square_size_m": float(cfg.square_size),
            "min_views": int(cfg.min_views),
            "undistort_alpha": float(cfg.undistort_alpha),
            "fix_k3": bool(cfg.fix_k3),
            "fix_aspect_ratio": bool(cfg.fix_aspect_ratio),
            "zero_tangent_dist": bool(cfg.zero_tangent_dist),
        },
        "result": result.report_dict(),
        "environment": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    if views is not None:
        payload["views"] = [
            {
                "source": v.source,
                "accepted": bool(v.accepted),
                "method": v.method,
                "corner_ratio": float(v.corner_ratio),
                "coverage_cells": int(v.coverage_cells),
                "error_px": None if v.error_px is None else float(v.error_px),
                "reason": v.reason,
            }
            for v in views
        ]
    if extra:
        payload.update(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def write_all_outputs(
    output_dir: str | Path,
    result: CalibrationResult,
    config: Optional[CalibrationConfig] = None,
    views: Optional[Sequence[ViewResult]] = None,
    extra_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """一次性写出四件套，返回 ``{名称: 路径}``。

    Args:
        output_dir: 输出目录（不存在会自动创建）。
        result: 标定结果。
        config: 标定配置。
        views: 所有视图（含被拒的），用于报告。
        extra_report: 追加进报告的字段（例如数据来源、相机型号）。

    Returns:
        ``{"ost": ..., "orbslam3": ..., "enhancer": ..., "report": ...}``
    """
    cfg = config or result.config or CalibrationConfig()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ost": write_ost_yaml(directory / "ost.yaml", result, cfg),
        "orbslam3": write_orbslam3_yaml(
            directory / f"{cfg.orbslam3_settings_name}.yaml", result, cfg
        ),
        "enhancer": write_enhancer_yaml(directory / "enhancer_camera_params.yaml", result, cfg),
        "report": write_report(directory / "calibration_report.json", result, cfg, extra_report, views),
    }
    return outputs
