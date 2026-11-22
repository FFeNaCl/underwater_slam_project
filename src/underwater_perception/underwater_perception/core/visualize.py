"""可视化工具：检测结果标注、掩码叠加、并排对比。

这些函数同时被 ROS 节点（发布 ``*/annotated`` 话题）和命令行工具（存图）使用，
所以放在 core 里而不是节点里，保证"命令行看到的"和"ROS 话题发布的"完全一致。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .params import CubeDetectorConfig
from .types import Detection

__all__ = ["annotate_detections", "mask_overlay", "side_by_side", "draw_status_panel"]

_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (60, 220, 255),
    (90, 200, 90),
    (255, 140, 90),
    (220, 120, 240),
    (120, 200, 240),
    (240, 220, 120),
)


def _color_for(index: int) -> Tuple[int, int, int]:
    return _PALETTE[index % len(_PALETTE)]


def annotate_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    config: Optional[CubeDetectorConfig] = None,
    mask: Optional[np.ndarray] = None,
    title: str = "",
    show_metrics: bool = True,
    show_score_terms: bool = False,
) -> np.ndarray:
    """在图像上画出检测轮廓、外接框、ID、置信度与关键几何指标。

    Args:
        image: BGR 图像。
        detections: 检测结果。
        config: 配置（用于取类名等展示信息）。
        mask: 可选的调色掩码；给出时会以半透明绿色叠加。
        title: 左上角标题栏文字。
        show_metrics: 是否绘制 fill/convexity/顶点数等指标。
        show_score_terms: 是否绘制评分分量（调参时开）。

    Returns:
        标注后的 BGR 图像（新数组，不修改输入）。
    """
    canvas = image.copy()
    if mask is not None:
        canvas = mask_overlay(canvas, mask)

    for index, detection in enumerate(detections):
        color = _color_for(detection.track_id if detection.track_id is not None else index)
        polygon = np.asarray(detection.polygon(), dtype=np.int32).reshape(-1, 1, 2)
        if len(polygon) >= 3:
            cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
            # 顶点编号：现场一眼能看出是 4/5/6 边形（立方体视角判断）
            for vertex_index, point in enumerate(polygon.reshape(-1, 2)):
                cv2.circle(canvas, tuple(int(v) for v in point), 3, color, -1)
                if show_metrics:
                    cv2.putText(
                        canvas,
                        str(vertex_index),
                        (int(point[0]) + 4, int(point[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        x1, y1, x2, y2 = (int(round(v)) for v in detection.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        label = f"{detection.class_name}"
        if detection.track_id is not None:
            label += f" #{detection.track_id}"
        label += f" {detection.confidence:.2f}"
        if detection.measurement is not None and detection.measurement.distance_m is not None:
            label += f" {detection.measurement.distance_m:.2f}m"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(canvas, (x1, top), (min(canvas.shape[1] - 1, x1 + tw + 8), top + th + baseline + 6), color, -1)
        cv2.putText(canvas, label, (x1 + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 25, 30), 2, cv2.LINE_AA)

        if show_metrics and detection.geometry is not None:
            geometry = detection.geometry
            lines = [
                f"V={geometry.vertex_count} fill={geometry.fill_ratio:.2f}",
                f"conv={geometry.convexity:.2f} dirs={geometry.edge_direction_count}",
                f"side={geometry.side_px:.0f}px purity={geometry.color_purity:.2f}",
            ]
            if show_score_terms and detection.score_terms:
                lines.append(" ".join(f"{k}={v:.2f}" for k, v in sorted(detection.score_terms.items())))
            base_y = min(canvas.shape[0] - 6, y2 + 16)
            for offset, text in enumerate(lines):
                cv2.putText(
                    canvas,
                    text,
                    (x1, base_y + offset * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    if title:
        canvas = draw_status_panel(canvas, [title])
    return canvas


def mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (60, 255, 120), alpha: float = 0.35) -> np.ndarray:
    """把二值掩码以半透明色叠加到图像上（调阈值时最直观的一步）。"""
    canvas = image.copy()
    if mask is None or mask.size == 0:
        return canvas
    if mask.shape[:2] != canvas.shape[:2]:
        mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(canvas)
    layer[mask > 0] = color
    return cv2.addWeighted(canvas, 1.0, layer, float(alpha), 0.0)


def side_by_side(left: np.ndarray, right: np.ndarray, left_title: str = "LEFT", right_title: str = "RIGHT") -> np.ndarray:
    """把两幅同高图像水平拼接并加标题（增强前后对比用）。"""
    height = min(left.shape[0], right.shape[0])
    left_resized = cv2.resize(left, (int(left.shape[1] * height / left.shape[0]), height)) if left.shape[0] != height else left
    right_resized = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]), height)) if right.shape[0] != height else right
    canvas = np.hstack([left_resized, right_resized])
    if left_title:
        cv2.putText(canvas, left_title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 230, 255), 2, cv2.LINE_AA)
    if right_title:
        cv2.putText(
            canvas,
            right_title,
            (left_resized.shape[1] + 12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 230, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_status_panel(image: np.ndarray, lines: Sequence[str], height: Optional[int] = None) -> np.ndarray:
    """在图像顶部画一条深色状态栏（节点里用来显示参数与统计）。"""
    canvas = image.copy()
    line_height = 18
    panel_height = height if height is not None else max(28, line_height * len(lines) + 10)
    panel = np.zeros((panel_height, canvas.shape[1], 3), dtype=np.uint8)
    panel[:] = (28, 34, 40)
    canvas[:panel_height] = cv2.addWeighted(canvas[:panel_height], 0.15, panel, 0.85, 0.0)
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, 16 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas
