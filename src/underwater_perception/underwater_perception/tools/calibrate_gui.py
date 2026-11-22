"""相机标定图形界面（标定板参数可随时修改）。

与 ``calibrate_cli`` 共用同一套算法与输出格式，区别只在交互方式：

* 左侧面板直接填写**标定板参数**：图案类型、内角点列数/行数、方格边长（毫米）；
* 数据来源支持 USB 相机实时采集、视频文件、图片文件夹；
* 实时采集窗口的按键：

  ==========  ==========================================
  ``SPACE``   采集当前帧（未检出棋盘或遮挡过多时不采）
  ``A``       自动采集开/关（按面板里的间隔尝试）
  ``R``       清空已采集视图
  ``C``       结束采集并立即标定
  ``Q``/``ESC`` 放弃本次采集
  ==========  ==========================================

运行（不需要 ROS）::

    python3 -m underwater_perception.tools.calibrate_gui
    ros2 run underwater_perception calibrate_gui

提示：**cols/rows 是内角点数**。8x6 内角点对应 9x7 个方格；
方格边长填毫米（25 mm 填 25），程序内部统一换算成米。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.calibration import (
    CalibrationResult,
    ViewResult,
    calibrate,
    undistort_image,
    view_from_array,
    view_from_path,
    write_all_outputs,
)
from ..core.camera_io import DEFAULT_FOURCC, DEFAULT_FPS, camera_info, open_camera
from ..core.params import CalibrationConfig

__all__ = [
    "SUPPORTED_PATTERNS",
    "board_summary",
    "validate_board_params",
    "build_calibration_config",
    "is_novel_view",
    "main",
]

SUPPORTED_PATTERNS: Tuple[str, ...] = ("chessboard", "circles")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")
WINDOW_NAME = "calibrate_gui"


# --------------------------------------------------------------------------- 纯函数（可单测）
def board_summary(pattern: str, cols: int, rows: int, square_mm: float) -> Dict[str, object]:
    """返回标定板的人话摘要（用于界面提示与自检）。

    Args:
        pattern: ``chessboard`` 或 ``circles``。
        cols: 内角点/圆点列数（沿 x）。
        rows: 内角点/圆点行数（沿 y）。
        square_mm: 方格边长或圆点间距（毫米）。

    Returns:
        含 ``inner_corners`` / ``squares`` / ``physical_mm`` / ``object_points`` 的字典。
    """
    cols_i, rows_i = int(cols), int(rows)
    square = float(square_mm)
    if pattern == "circles":
        # 圆点阵列没有"多一格"的问题，物理跨度是 (n-1) 个间距
        squares = f"{cols_i}x{rows_i}"
        physical = (max(cols_i - 1, 0) * square, max(rows_i - 1, 0) * square)
    else:
        # 棋盘格：8x6 内角点 = 9x7 个方格，物理尺寸为 9*边长 x 7*边长
        squares = f"{cols_i + 1}x{rows_i + 1}"
        physical = ((cols_i + 1) * square, (rows_i + 1) * square)
    return {
        "pattern": pattern,
        "inner_corners": f"{cols_i}x{rows_i}",
        "squares": squares,
        "square_mm": square,
        "physical_mm": physical,
        "object_points": cols_i * rows_i,
    }


def validate_board_params(pattern: str, cols: int, rows: int, square_mm: float) -> Tuple[List[str], List[str]]:
    """检查标定板参数，返回 ``(errors, warnings)``。

    只拦"一定标不出来"的错误：图案不支持、行列数太小、边长为非正数。
    其他情况给警告，不阻止用户继续。
    """
    errors: List[str] = []
    warnings: List[str] = []
    if pattern not in SUPPORTED_PATTERNS:
        errors.append(f"图形界面暂不支持图案 {pattern!r}，请选 {SUPPORTED_PATTERNS}")
    try:
        cols_i, rows_i = int(cols), int(rows)
    except (TypeError, ValueError):
        return ["列数/行数必须是整数"], warnings
    if cols_i < 3 or rows_i < 3:
        errors.append("内角点列数和行数都至少为 3（推荐 6 以上）")
    if cols_i * rows_i < 9:
        errors.append("内角点总数至少 9 个，否则标定会严重病态")
    try:
        square = float(square_mm)
    except (TypeError, ValueError):
        return errors + ["方格边长必须是数字（单位毫米）"], warnings
    if not np.isfinite(square) or square <= 0.0:
        errors.append("方格边长必须大于 0（单位毫米）")
    elif square < 1.0:
        # 最常见的手误：把 25 mm 写成了 0.025（米）
        errors.append(
            f"方格边长单位是毫米，当前填的是 {square:g}，看起来是按米填的；"
            f"25 mm 请填 25（不是 0.025）"
        )
    elif square < 5.0:
        warnings.append("方格边长小于 5 mm：打印误差和角点定位误差占比会明显变大")
    if cols_i > 20 or rows_i > 20:
        warnings.append("内角点数超过 20：需要很高分辨率才能稳定检出，通常没必要")
    if cols_i * rows_i > 400:
        warnings.append("内角点过多：检测和标定会变慢，且容易漏检")
    return errors, warnings


def build_calibration_config(
    pattern: str,
    cols: int,
    rows: int,
    square_mm: float,
    output_dir: str,
    camera_name: str,
    orbslam3_name: str,
    fps: float,
    min_views: int,
    max_views: int,
    fix_k3: bool = False,
    fix_aspect_ratio: bool = False,
    zero_tangent_dist: bool = False,
    undistort_alpha: float = 0.0,
) -> CalibrationConfig:
    """把界面参数转成 :class:`CalibrationConfig`（毫米 -> 米只在这里发生）。"""
    return CalibrationConfig(
        pattern=str(pattern),
        cols=int(cols),
        rows=int(rows),
        square_size=float(square_mm) / 1000.0,
        min_views=max(3, int(min_views)),
        max_views=max(3, int(max_views)),
        min_corners_ratio=0.85,
        coverage_grid=3,
        min_coverage_cells=3,
        subpix_window=11,
        fix_k3=bool(fix_k3),
        fix_aspect_ratio=bool(fix_aspect_ratio),
        zero_tangent_dist=bool(zero_tangent_dist),
        undistort_alpha=float(undistort_alpha),
        camera_name=str(camera_name),
        output_dir=str(output_dir),
        orbslam3_settings_name=str(orbslam3_name),
        fps=float(fps),
        save_undistorted=True,
        save_debug_corners=True,
    )


def _view_signature(view: ViewResult) -> Optional[Tuple[np.ndarray, float]]:
    """视图的粗略几何签名：角点中心 + 平均半径（用于自动采集去重）。"""
    if not view.found:
        return None
    points = np.asarray(view.corners, dtype=np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    spread = float(np.linalg.norm(points - center, axis=1).mean())
    return center, spread


def is_novel_view(
    view: ViewResult,
    accepted: Sequence[ViewResult],
    min_shift_px: float = 12.0,
    min_scale_ratio: float = 0.06,
) -> bool:
    """判断新视图是否与最近已采集视图"足够不同"（自动采集时抑制重复帧）。

    只比较角点中心与点阵尺度：板子只原地旋转但中心不变时仍可能被判重复，
    因此手动采集（``SPACE``）不受此限制。
    """
    signature = _view_signature(view)
    if signature is None:
        return False
    center, spread = signature
    for previous in list(accepted)[-5:]:
        other = _view_signature(previous)
        if other is None:
            continue
        other_center, other_spread = other
        shift = float(np.linalg.norm(center - other_center))
        scale_delta = abs(spread - other_spread) / max(other_spread, 1.0)
        if shift < float(min_shift_px) and scale_delta < float(min_scale_ratio):
            return False
    return True


def _format_result(result: CalibrationResult) -> str:
    """把标定结果格式化成界面日志文本。"""
    if not result.ok:
        return f"标定失败: {result.message}"
    lines = [
        f"有效视图      : {result.n_views_used}/{result.n_views_total}",
        f"RMS 重投影误差: {result.rms:.4f} px（<0.5 优秀，>1.0 需检查）",
        f"内参 fx,fy    : {result.fx:.3f}, {result.fy:.3f} px",
        f"主点 cx,cy    : {result.cx:.3f}, {result.cy:.3f} px",
        "畸变 k1,k2,p1,p2,k3: " + ", ".join(f"{v:.5f}" for v in result.distortion_list(5)),
        f"质量等级      : {result.quality_grade()}",
    ]
    worst = sorted(result.per_view_errors, reverse=True)[:3]
    if worst:
        lines.append("误差最大的 3 个视图: " + ", ".join(f"{value:.3f} px" for value in worst))
    for warning in result.warnings:
        lines.append(f"[警告] {warning}")
    return "\n".join(lines)


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """读取可能含中文/空格的路径（Windows 上 OpenCV 的经典问题）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except OSError:
        return None


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """写入可能含中文/空格的路径。"""
    suffix = path.suffix if path.suffix else ".jpg"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        buffer.tofile(str(path))
        return True
    except OSError:
        return False


def _open_directory(path: Path) -> None:
    """跨平台打开目录（失败时静默，日志里已有路径）。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


# --------------------------------------------------------------------------- GUI
class CalibrationApp:
    """Tkinter 界面 + OpenCV 采集窗口。"""

    def __init__(self, root, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title("水下单目相机标定工具 - 标定板参数可修改")
        self.root.geometry("1080x760")
        self.views: List[ViewResult] = []
        self.last_result: Optional[CalibrationResult] = None
        self._build_variables(args)
        self._build_ui()
        self._update_board_summary()

    # ------------------------------------------------------------------ 变量
    def _build_variables(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import ttk  # noqa: F401  (确保子模块可用)

        self.pattern_var = tk.StringVar(value=getattr(args, "pattern", "chessboard"))
        self.cols_var = tk.IntVar(value=int(getattr(args, "cols", 8)))
        self.rows_var = tk.IntVar(value=int(getattr(args, "rows", 6)))
        self.square_mm_var = tk.DoubleVar(value=float(getattr(args, "square_mm", 25.0)))
        self.min_views_var = tk.IntVar(value=int(getattr(args, "min_views", 12)))
        self.max_views_var = tk.IntVar(value=int(getattr(args, "max_views", 40)))
        self.capture_target_var = tk.IntVar(value=int(getattr(args, "capture", 25)))
        self.interval_var = tk.DoubleVar(value=float(getattr(args, "interval", 1.0)))
        self.sample_every_var = tk.IntVar(value=int(getattr(args, "sample_every", 10)))

        self.source_var = tk.StringVar(value="images" if getattr(args, "images", "") else "camera")
        self.camera_index_var = tk.IntVar(value=int(getattr(args, "camera", 0)))
        self.resolution_var = tk.StringVar(value=str(getattr(args, "resolution", "1280x720")))
        self.fourcc_var = tk.StringVar(value=str(getattr(args, "fourcc", DEFAULT_FOURCC)))
        self.camera_fps_var = tk.DoubleVar(value=float(getattr(args, "camera_fps", DEFAULT_FPS)))
        self.video_path_var = tk.StringVar(value=str(getattr(args, "video", "")))
        self.images_dir_var = tk.StringVar(value=str(getattr(args, "images", "")))

        self.out_dir_var = tk.StringVar(
            value=str(getattr(args, "out", "") or (Path.cwd() / "calibration_out"))
        )
        self.camera_name_var = tk.StringVar(value=str(getattr(args, "camera_name", "water_camera")))
        self.orbslam_name_var = tk.StringVar(value=str(getattr(args, "orbslam3_name", "water_1280x480")))
        self.fps_var = tk.DoubleVar(value=float(getattr(args, "fps", 30.0)))
        self.fix_k3_var = tk.BooleanVar(value=False)
        self.fix_aspect_var = tk.BooleanVar(value=False)
        self.zero_tangent_var = tk.BooleanVar(value=False)
        self.alpha_var = tk.DoubleVar(value=0.0)

    # ------------------------------------------------------------------ 布局
    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        # ---- 标定板 ----
        board = ttk.LabelFrame(left, text="1. 标定板参数（可修改）", padding=8)
        board.pack(fill=tk.X, pady=(0, 8))
        row = 0
        ttk.Label(board, text="图案类型").grid(row=row, column=0, sticky="w", pady=2)
        pattern_box = ttk.Combobox(
            board,
            textvariable=self.pattern_var,
            values=list(SUPPORTED_PATTERNS),
            state="readonly",
            width=14,
        )
        pattern_box.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        ttk.Label(board, text="内角点列数 (x)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(board, from_=3, to=30, textvariable=self.cols_var, width=16).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        ttk.Label(board, text="内角点行数 (y)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(board, from_=3, to=30, textvariable=self.rows_var, width=16).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        ttk.Label(board, text="方格边长 (毫米)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            board, from_=1.0, to=500.0, increment=0.5, textvariable=self.square_mm_var, width=16
        ).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        self.board_summary_label = ttk.Label(
            board, text="", justify=tk.LEFT, wraplength=260, foreground="#154c79"
        )
        self.board_summary_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        board.columnconfigure(1, weight=1)

        for variable in (self.pattern_var, self.cols_var, self.rows_var, self.square_mm_var):
            variable.trace_add("write", lambda *_: self._update_board_summary())

        # ---- 采集 ----
        capture = ttk.LabelFrame(left, text="2. 采集与标定参数", padding=8)
        capture.pack(fill=tk.X, pady=(0, 8))
        row = 0
        ttk.Label(capture, text="最少有效视图").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(capture, from_=3, to=100, textvariable=self.min_views_var, width=16).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        ttk.Label(capture, text="最多保留视图").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(capture, from_=3, to=200, textvariable=self.max_views_var, width=16).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        ttk.Label(capture, text="采集目标张数").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(capture, from_=3, to=200, textvariable=self.capture_target_var, width=16).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        ttk.Label(capture, text="自动采集间隔 (秒)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            capture, from_=0.2, to=10.0, increment=0.2, textvariable=self.interval_var, width=16
        ).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        ttk.Checkbutton(capture, text="固定 k3=0", variable=self.fix_k3_var).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Checkbutton(capture, text="固定 fx/fy 宽高比", variable=self.fix_aspect_var).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Checkbutton(capture, text="切向畸变置零", variable=self.zero_tangent_var).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(capture, text="去畸变保留比例 alpha").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            capture, from_=0.0, to=1.0, increment=0.1, textvariable=self.alpha_var, width=16
        ).grid(row=row, column=1, sticky="ew", pady=2)
        capture.columnconfigure(1, weight=1)

        # ---- 数据来源 ----
        source = ttk.LabelFrame(left, text="3. 数据来源", padding=8)
        source.pack(fill=tk.X, pady=(0, 8))
        row = 0
        ttk.Radiobutton(source, text="USB/CSI 相机", value="camera", variable=self.source_var).grid(
            row=row, column=0, sticky="w", columnspan=3
        )
        row += 1
        ttk.Label(source, text="相机序号").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(source, from_=0, to=16, textvariable=self.camera_index_var, width=6).grid(
            row=row, column=1, sticky="w", padx=4
        )
        ttk.Label(source, text="分辨率").grid(row=row, column=2, sticky="w")
        ttk.Entry(source, textvariable=self.resolution_var, width=11).grid(row=row, column=3, sticky="w")
        row += 1
        ttk.Label(source, text="像素格式").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            source,
            textvariable=self.fourcc_var,
            values=["MJPG", "YUYV"],
            state="readonly",
            width=6,
        ).grid(row=row, column=1, sticky="w", padx=4)
        ttk.Label(source, text="相机 FPS").grid(row=row, column=2, sticky="w")
        ttk.Spinbox(source, from_=1.0, to=120.0, textvariable=self.camera_fps_var, width=11).grid(
            row=row, column=3, sticky="w"
        )
        row += 1
        ttk.Radiobutton(source, text="视频文件", value="video", variable=self.source_var).grid(
            row=row, column=0, sticky="w", columnspan=3
        )
        row += 1
        ttk.Entry(source, textvariable=self.video_path_var, width=30).grid(
            row=row, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(source, text="浏览", command=self._browse_video, width=6).grid(row=row, column=3)
        row += 1
        ttk.Label(source, text="抽帧间隔").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(source, from_=1, to=200, textvariable=self.sample_every_var, width=6).grid(
            row=row, column=1, sticky="w", padx=4
        )
        ttk.Label(source, text="帧").grid(row=row, column=2, sticky="w")
        row += 1
        ttk.Radiobutton(source, text="图片文件夹", value="images", variable=self.source_var).grid(
            row=row, column=0, sticky="w", columnspan=3
        )
        row += 1
        ttk.Entry(source, textvariable=self.images_dir_var, width=30).grid(
            row=row, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(source, text="浏览", command=self._browse_images, width=6).grid(row=row, column=3)
        source.columnconfigure(2, weight=1)

        # ---- 输出 ----
        output = ttk.LabelFrame(left, text="4. 输出", padding=8)
        output.pack(fill=tk.X)
        row = 0
        ttk.Label(output, text="输出目录").grid(row=row, column=0, sticky="w")
        ttk.Entry(output, textvariable=self.out_dir_var, width=30).grid(
            row=row, column=1, columnspan=2, sticky="ew"
        )
        ttk.Button(output, text="浏览", command=self._browse_out, width=6).grid(row=row, column=3)
        row += 1
        ttk.Label(output, text="相机名称").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(output, textvariable=self.camera_name_var, width=18).grid(
            row=row, column=1, columnspan=3, sticky="ew"
        )
        row += 1
        ttk.Label(output, text="ORB-SLAM3 设置名").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(output, textvariable=self.orbslam_name_var, width=18).grid(
            row=row, column=1, columnspan=3, sticky="ew"
        )
        row += 1
        ttk.Label(output, text="Camera.fps").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(output, from_=1.0, to=120.0, textvariable=self.fps_var, width=18).grid(
            row=row, column=1, columnspan=3, sticky="ew"
        )
        output.columnconfigure(2, weight=1)

        # ---- 右侧按钮与日志 ----
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        buttons = ttk.Frame(right)
        buttons.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(buttons, text="校验参数", command=self._check_params).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="开始采集并标定", command=self._start_capture).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="仅用图片文件夹标定", command=self._calibrate_images).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(buttons, text="打开输出目录", command=self._open_output).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="清空日志", command=self._clear_log).pack(side=tk.LEFT)

        hint = ttk.Label(
            right,
            text=(
                "采集窗口按键：SPACE=采集  A=自动采集  R=清空  C=标定  Q/ESC=放弃。"
                " 标定板/相机都要在水下，水质与正式实验一致。"
            ),
            wraplength=700,
            foreground="#555555",
        )
        hint.grid(row=1, column=0, sticky="w", pady=(0, 6))

        log_frame = ttk.LabelFrame(right, text="日志", padding=6)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word", height=24, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------ 界面辅助
    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _update_board_summary(self) -> None:
        try:
            info = board_summary(
                self.pattern_var.get(), self.cols_var.get(), self.rows_var.get(), self.square_mm_var.get()
            )
        except Exception:
            self.board_summary_label.configure(text="参数格式不正确")
            return
        width, height = info["physical_mm"]
        self.board_summary_label.configure(
            text=(
                f"内角点 {info['inner_corners']}；方格 {info['squares']}；"
                f"物理尺寸约 {width:.1f} x {height:.1f} mm；"
                f"有效点 {info['object_points']} 个"
            )
        )

    def _config_from_ui(self) -> Optional[CalibrationConfig]:
        errors, warnings = validate_board_params(
            self.pattern_var.get(),
            self.cols_var.get(),
            self.rows_var.get(),
            self.square_mm_var.get(),
        )
        for warning in warnings:
            self._log(f"[提示] {warning}")
        if errors:
            for error in errors:
                self._log(f"[参数错误] {error}")
            return None
        return build_calibration_config(
            pattern=self.pattern_var.get(),
            cols=self.cols_var.get(),
            rows=self.rows_var.get(),
            square_mm=self.square_mm_var.get(),
            output_dir=self.out_dir_var.get(),
            camera_name=self.camera_name_var.get(),
            orbslam3_name=self.orbslam_name_var.get(),
            fps=self.fps_var.get(),
            min_views=self.min_views_var.get(),
            max_views=self.max_views_var.get(),
            fix_k3=self.fix_k3_var.get(),
            fix_aspect_ratio=self.fix_aspect_var.get(),
            zero_tangent_dist=self.zero_tangent_var.get(),
            undistort_alpha=self.alpha_var.get(),
        )

    def _check_params(self) -> None:
        config = self._config_from_ui()
        if config is None:
            return
        info = board_summary(config.pattern, config.cols, config.rows, config.square_size * 1000.0)
        self._log("参数检查通过。")
        self._log(
            "  图案=%s 内角点=%s 方格=%s 方格边长=%.2f mm 物理尺寸=%.1fx%.1f mm"
            % (
                config.pattern,
                info["inner_corners"],
                info["squares"],
                config.square_size * 1000.0,
                info["physical_mm"][0],
                info["physical_mm"][1],
            )
        )
        self._log("  输出目录: %s" % config.output_dir)
        self._log("  提示：打印后务必用卡尺量方格边长，打印缩放会改变真实尺寸。")

    def _browse_video(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="选择标定视频",
            filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")],
        )
        if path:
            self.video_path_var.set(path)
            self.source_var.set("video")

    def _browse_images(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="选择标定图片文件夹")
        if path:
            self.images_dir_var.set(path)
            self.source_var.set("images")

    def _browse_out(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.out_dir_var.set(path)

    def _open_output(self) -> None:
        _open_directory(Path(self.out_dir_var.get()).expanduser())

    # ------------------------------------------------------------------ 采集入口
    def _start_capture(self) -> None:
        config = self._config_from_ui()
        if config is None:
            return
        source = self.source_var.get()
        if source == "camera":
            views = self._capture_camera(config)
        elif source == "video":
            views = self._collect_video(config)
        else:
            views = self._collect_images(config)
        if views:
            self._calibrate(views, config)

    def _calibrate_images(self) -> None:
        self.source_var.set("images")
        config = self._config_from_ui()
        if config is None:
            return
        views = self._collect_images(config)
        if views:
            self._calibrate(views, config)

    def _parse_resolution(self) -> Tuple[int, int]:
        text = self.resolution_var.get().strip().lower().replace(" ", "")
        try:
            width, height = text.split("x")
            return int(width), int(height)
        except ValueError:
            return 0, 0

    def _capture_camera(self, config: CalibrationConfig) -> List[ViewResult]:
        """实时相机采集；返回所有尝试过的视图（含未接受的）。"""
        width, height = self._parse_resolution()
        capture = open_camera(
            self.camera_index_var.get(),
            width=width,
            height=height,
            fourcc=self.fourcc_var.get(),
            fps=self.camera_fps_var.get(),
        )
        if not capture.isOpened():
            self._log(
                f"[错误] 无法打开相机 {self.camera_index_var.get()}（检查 /dev/video* 权限与占用）。"
                " Linux 下 /dev/video1、/dev/video3 常是 metadata 节点，请改用 0/2。"
            )
            return []
        info = camera_info(capture)
        self._log(
            "相机实际格式: {fourcc} {width}x{height} @ {fps:.1f} fps".format(**info)
        )
        if info.get("fourcc") != "MJPG" and width >= 1280:
            self._log("[警告] 当前不是 MJPG，1280x720 的 YUYV 在 USB2 上通常只有 5~10 fps。")

        views: List[ViewResult] = []
        auto = False
        last_capture = 0.0
        self.root.withdraw()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._log("相机已打开：SPACE 采集，A 自动采集，R 清空，C 标定，Q 退出。")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    self._log("[错误] 相机读帧失败")
                    break
                view = view_from_array(frame, config, source=f"camera{self.camera_index_var.get()}#{len(views)}")
                preview = frame.copy()
                if view.found:
                    cv2.drawChessboardCorners(preview, config.pattern_size, view.corners, view.accepted)

                overlay = (
                    f"board {config.cols}x{config.rows} | square {config.square_size*1000:.1f}mm | "
                    f"views {len(views)}/{int(self.capture_target_var.get())} | auto {'ON' if auto else 'OFF'}"
                )
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 62), (18, 32, 42), -1)
                cv2.putText(preview, overlay, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 240, 245), 1, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    "SPACE capture | A auto | R reset | C calibrate | Q quit",
                    (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (80, 210, 255),
                    1,
                    cv2.LINE_AA,
                )
                if view.found and not view.accepted:
                    cv2.putText(
                        preview,
                        f"corners {view.corner_ratio*100:.0f}% - board not fully visible",
                        (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (60, 60, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                now = time.time()
                if key in (ord("q"), 27):
                    self._log("已放弃本次采集。")
                    views = []
                    break
                if key == ord("r"):
                    views.clear()
                    self._log("已清空采集视图。")
                    continue
                if key == ord("a"):
                    auto = not auto
                    self._log(f"自动采集 {'开启' if auto else '关闭'}（间隔 {self.interval_var.get():.1f} s）")
                    continue
                if key == ord("c"):
                    break

                should_capture = False
                if key == ord(" "):
                    if view.accepted:
                        should_capture = True
                    else:
                        self._log("当前帧未检出完整标定板，未采集。")
                elif auto and view.accepted and (now - last_capture) >= float(self.interval_var.get()):
                    if is_novel_view(view, views):
                        should_capture = True
                    else:
                        self._log("自动采集：与最近视图过于相似，跳过。")

                if should_capture:
                    views.append(view)
                    last_capture = now
                    self._log(
                        f"采集 {len(views)}/{int(self.capture_target_var.get())}："
                        f"角点占比 {view.corner_ratio*100:.0f}%"
                    )
                    if len(views) >= int(self.capture_target_var.get()):
                        auto = False
                        self._log("已达到采集目标张数，自动采集已关闭；按 C 结束并标定。")
                        continue

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self._log("采集窗口被关闭。")
                    break
        finally:
            capture.release()
            cv2.destroyWindow(WINDOW_NAME)
            self.root.deiconify()
        return views

    def _collect_video(self, config: CalibrationConfig) -> List[ViewResult]:
        path = Path(self.video_path_var.get()).expanduser()
        if not path.is_file():
            self._log(f"[错误] 视频不存在: {path}")
            return []
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self._log(f"[错误] 无法打开视频: {path}")
            return []
        step = max(1, int(self.sample_every_var.get()))
        target = max(3, int(self.capture_target_var.get()))
        views: List[ViewResult] = []
        index = 0
        self._log(f"读取视频 {path}，每 {step} 帧取一帧，目标 {target} 个有效视图。")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % step == 0:
                    view = view_from_array(frame, config, source=f"{path.name}#{index}")
                    if view.accepted:
                        views.append(view)
                        if len(views) % 5 == 0:
                            self._log(f"  已得到 {len(views)} 个有效视图（帧 {index}）")
                        if len(views) >= min(target, int(config.max_views)):
                            break
                index += 1
                if index % 300 == 0:
                    self.root.update_idletasks()
        finally:
            capture.release()
        self._log(f"视频共 {index} 帧，得到 {len(views)} 个有效视图。")
        return views

    def _collect_images(self, config: CalibrationConfig) -> List[ViewResult]:
        directory = Path(self.images_dir_var.get()).expanduser()
        if not directory.is_dir():
            self._log(f"[错误] 图片目录不存在: {directory}")
            return []
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            self._log(f"[错误] 目录里没有图片: {directory}")
            return []
        self._log(f"从 {directory} 读取 {len(paths)} 张图片……")
        views: List[ViewResult] = []
        for count, path in enumerate(paths, start=1):
            view = view_from_path(path, config)
            views.append(view)
            if view.accepted:
                self._log(f"  [{count}/{len(paths)}] 检出 {path.name}（角点占比 {view.corner_ratio*100:.0f}%）")
            if count % 20 == 0:
                self.root.update_idletasks()
        rejected = sum(1 for view in views if not view.accepted)
        self._log(f"图片读取完成：有效 {len(views) - rejected}，未通过 {rejected}。")
        return views

    # ------------------------------------------------------------------ 标定与输出
    def _calibrate(self, views: Sequence[ViewResult], config: CalibrationConfig) -> None:
        accepted = [view for view in views if view.accepted and view.found]
        if len(accepted) < 3:
            self._log("[错误] 有效视图不足 3 个，无法标定。请检查标定板参数是否与实物一致。")
            return
        image_size = accepted[0].image_size
        mismatched = [view for view in accepted if view.image_size != image_size]
        if mismatched:
            self._log(f"[警告] 有 {len(mismatched)} 个视图尺寸与首个视图不同，已忽略。")
            accepted = [view for view in accepted if view.image_size == image_size]
        if len(accepted) < 3:
            self._log("[错误] 同尺寸有效视图不足 3 个。")
            return

        self._log("开始标定……")
        self.root.update_idletasks()
        result = calibrate(accepted, image_size, config, source="gui")
        self.last_result = result
        self._log(_format_result(result))
        if not result.ok:
            return

        outputs = write_all_outputs(config.output_dir, result, config, views=views)
        for name, path in outputs.items():
            self._log(f"  {name:9s} -> {path}")

        # 保存几张去畸变示例，方便人眼检查边缘是否变直
        out_dir = Path(config.output_dir) / "undistorted"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for index, view in enumerate(accepted[:5]):
            if not isinstance(view.source, str) or not Path(view.source).is_file():
                continue
            image = _imread_unicode(Path(view.source))
            if image is None:
                continue
            undistorted, _, _ = undistort_image(image, result, config.undistort_alpha)
            if _imwrite_unicode(out_dir / f"undistorted_{index:02d}.jpg", undistorted):
                saved += 1
        if saved:
            self._log(f"去畸变示例图: {out_dir}（{saved} 张，打开检查直线是否变直）")

        self._log("")
        self._log("下一步：")
        self._log(f"  1) 把 {Path(config.output_dir) / (config.orbslam3_settings_name + '.yaml')} 复制到")
        self._log("     src/ros2_orb_slam3/orb_slam3/config/Monocular/")
        self._log(f"  2) 把 {Path(config.output_dir) / 'enhancer_camera_params.yaml'} 里的")
        self._log("     camera_matrix / distortion_coeffs 填进 config/enhancer_underwater.yaml")
        self._log("     并设置 use_undistort: true")
        self._log("  3) 重启增强/SLAM 链路，用 ros2 topic echo /slam/status --once 确认跟踪状态")


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口（GUI）。"""
    parser = argparse.ArgumentParser(description="图形化相机标定工具（标定板参数可修改）")
    parser.add_argument("--pattern", default="chessboard", choices=list(SUPPORTED_PATTERNS))
    parser.add_argument("--cols", type=int, default=8, help="内角点列数")
    parser.add_argument("--rows", type=int, default=6, help="内角点行数")
    parser.add_argument("--square-mm", type=float, default=25.0, help="方格边长（毫米）")
    parser.add_argument("--camera", type=int, default=0, help="USB/CSI 相机序号（video1/video3 常是 metadata，别用）")
    parser.add_argument("--resolution", default="1280x720", help="相机分辨率，例如 1280x720")
    parser.add_argument("--fourcc", default=DEFAULT_FOURCC, choices=["MJPG", "YUYV"], help="像素格式，MJPG 才有 30 fps")
    parser.add_argument("--camera-fps", type=float, default=DEFAULT_FPS, help="相机目标帧率")
    parser.add_argument("--video", default="", help="视频文件路径")
    parser.add_argument("--images", default="", help="图片文件夹")
    parser.add_argument("--out", default="", help="输出目录")
    parser.add_argument("--camera-name", default="water_camera")
    parser.add_argument("--orbslam3-name", default="water_1280x480")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-views", type=int, default=12)
    parser.add_argument("--max-views", type=int, default=40)
    parser.add_argument("--capture", type=int, default=25, help="相机模式目标采集张数")
    parser.add_argument("--interval", type=float, default=1.0, help="自动采集间隔（秒）")
    parser.add_argument("--sample-every", type=int, default=10, help="视频模式抽帧间隔")
    args, _unknown = parser.parse_known_args(argv)

    try:
        import tkinter as tk
    except ImportError:
        print("缺少 tkinter，请先安装：sudo apt install python3-tk", file=sys.stderr)
        return 3
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法启动图形界面（没有 DISPLAY？）: {exc}", file=sys.stderr)
        print("无显示器时请改用：python3 -m underwater_perception.tools.calibrate_cli --images <目录>", file=sys.stderr)
        return 3

    CalibrationApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
