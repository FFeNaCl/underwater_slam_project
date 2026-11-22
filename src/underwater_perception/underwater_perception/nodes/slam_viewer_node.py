"""SLAM 建图可视化窗口（OpenCV 2D 俯视图 + 状态面板）。

RViz 是主可视化工具，但有些机器上 RViz/OpenGL 不方便；这个节点只依赖 OpenCV，
把 ``/slam/path``、``/slam/map_points``、``/semantic/markers`` 和 ``/slam/status``
画成一张实时俯视图：

* 绿色折线：相机轨迹（世界系 X-Z 平面）
* 灰色点：ORB-SLAM3 稀疏地图点
* 绿/黄圆点：语义地标（绿=triangulated，黄=scale_fallback）
* 左上角：跟踪状态、地图点数、FPS、轨迹长度

运行::

    ros2 run underwater_perception slam_viewer

可选参数：``path_topic``、``points_topic``、``status_topic``、``landmarks_topic``、``window_name``。
"""

from __future__ import annotations

import json
import math
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from .image_bridge import msg_to_bgr

CANVAS = 900
MARGIN = 70


class SlamViewerNode(Node):
    """把 SLAM 话题画成 2D 俯视图。"""

    def __init__(self) -> None:
        super().__init__("slam_viewer")

        self.path_topic = str(self.declare_parameter("path_topic", "/slam/path").value)
        self.points_topic = str(self.declare_parameter("points_topic", "/slam/map_points").value)
        self.status_topic = str(self.declare_parameter("status_topic", "/slam/status").value)
        self.landmarks_topic = str(self.declare_parameter("landmarks_topic", "/semantic/landmarks").value)
        self.image_topic = str(self.declare_parameter("image_topic", "/image_enhanced").value)
        self.window_name = str(self.declare_parameter("window_name", "SLAM Map (top view X-Z)").value)
        self.update_hz = float(self.declare_parameter("update_hz", 5.0).value)
        self.max_points = int(self.declare_parameter("max_points", 6000).value)
        self.latest_image = None

        self.trajectory: List[Tuple[float, float]] = []
        self.points: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.landmarks: List[dict] = []
        self.status: dict = {}
        self._closed = False

        self.create_subscription(Path, self.path_topic, self.on_path, 10)
        self.create_subscription(PointCloud2, self.points_topic, self.on_points, 10)
        self.create_subscription(String, self.status_topic, self.on_status, 10)
        self.create_subscription(String, self.landmarks_topic, self.on_landmarks, 10)
        self.create_subscription(Image, self.image_topic, self.on_image, 5)
        self.create_timer(1.0 / max(self.update_hz, 0.5), self.on_timer)

        self.get_logger().info(
            f"SLAM 可视化启动: {self.path_topic} + {self.points_topic} -> OpenCV 窗口"
        )
        self.get_logger().info("窗口按键: q 或 ESC 关闭（不会停止 SLAM 节点）")

    # ------------------------------------------------------------------ 回调
    def on_path(self, msg: Path) -> None:
        """保存轨迹的 X-Z 投影（世界系）。"""
        self.trajectory = [
            (float(pose.pose.position.x), float(pose.pose.position.z)) for pose in msg.poses
        ]

    def on_points(self, msg: PointCloud2) -> None:
        """保存地图点（X-Z 投影）。"""
        try:
            values = list(
                point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            )
        except Exception:
            return
        if not values:
            self.points = np.zeros((0, 3), dtype=np.float64)
            return
        # read_points 在不同 Humble 小版本里可能返回 tuple 生成器或结构化数组，
        # 统一转成 (N, 3) 浮点数组。
        array = np.array(
            [(float(item[0]), float(item[1]), float(item[2])) for item in values],
            dtype=np.float64,
        )
        if self.max_points > 0 and len(array) > self.max_points:
            step = max(1, len(array) // self.max_points)
            array = array[::step][: self.max_points]
        self.points = array

    def on_status(self, msg: String) -> None:
        try:
            self.status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def on_landmarks(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.landmarks = list(payload.get("landmarks", []))
        except json.JSONDecodeError:
            pass

    def on_image(self, msg: Image) -> None:
        """保存最新相机图，供窗口右下角预览。"""
        try:
            self.latest_image = msg_to_bgr(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------ 绘制
    def _bounds(self) -> Tuple[float, float, float, float]:
        xs: List[float] = [p[0] for p in self.trajectory]
        zs: List[float] = [p[1] for p in self.trajectory]
        if len(self.points):
            xs.extend(self.points[:, 0].tolist())
            zs.extend(self.points[:, 2].tolist())
        for landmark in self.landmarks:
            position = landmark.get("position", [0.0, 0.0, 0.0])
            if len(position) >= 3:
                xs.append(float(position[0]))
                zs.append(float(position[2]))
        if not xs:
            return -1.0, 1.0, -1.0, 1.0
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
        span = max(x_max - x_min, z_max - z_min, 0.5)
        pad = 0.15 * span
        return x_min - pad, x_max + pad, z_min - pad, z_max + pad

    def _render(self) -> np.ndarray:
        canvas = np.full((CANVAS, CANVAS, 3), (28, 32, 38), dtype=np.uint8)
        x_min, x_max, z_min, z_max = self._bounds()
        span_x = max(x_max - x_min, 1e-6)
        span_z = max(z_max - z_min, 1e-6)
        scale = min((CANVAS - 2 * MARGIN) / span_x, (CANVAS - 2 * MARGIN) / span_z)

        def to_px(x: float, z: float) -> Tuple[int, int]:
            px = int(MARGIN + (x - x_min) * scale)
            py = int(CANVAS - MARGIN - (z - z_min) * scale)
            return px, py

        # 网格
        for index in range(0, CANVAS, 100):
            cv2.line(canvas, (index, 0), (index, CANVAS), (45, 52, 62), 1)
            cv2.line(canvas, (0, index), (CANVAS, index), (45, 52, 62), 1)

        # 地图点
        for x, _y, z in self.points:
            px, py = to_px(float(x), float(z))
            if 0 <= px < CANVAS and 0 <= py < CANVAS:
                cv2.circle(canvas, (px, py), 2, (190, 190, 190), -1, cv2.LINE_AA)

        # 轨迹
        if len(self.trajectory) >= 2:
            pts = np.array([to_px(x, z) for x, z in self.trajectory], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (60, 220, 120), 2, cv2.LINE_AA)
        if self.trajectory:
            px, py = to_px(*self.trajectory[-1])
            cv2.circle(canvas, (px, py), 6, (40, 90, 255), -1, cv2.LINE_AA)

        # 语义地标
        for landmark in self.landmarks:
            position = landmark.get("position", [0.0, 0.0, 0.0])
            if len(position) < 3:
                continue
            px, py = to_px(float(position[0]), float(position[2]))
            color = (60, 220, 120) if landmark.get("source") == "triangulated" else (60, 200, 240)
            cv2.circle(canvas, (px, py), 7, color, -1, cv2.LINE_AA)
            label = f"L{landmark.get('landmark_id', '?')} {landmark.get('class_name', '')}"
            cv2.putText(canvas, label, (px + 9, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 状态面板
        state = self.status.get("tracking_state", "NO_DATA")
        n_points = self.status.get("n_map_points", len(self.points))
        fps = self.status.get("fps", 0.0)
        path_len = 0.0
        for (x0, z0), (x1, z1) in zip(self.trajectory[:-1], self.trajectory[1:]):
            path_len += math.hypot(x1 - x0, z1 - z0)
        lines = [
            f"state: {state}",
            f"map points: {n_points}",
            f"path length: {path_len:.3f} (map units)",
            f"fps: {fps:.1f}",
            f"trajectory: {len(self.trajectory)} poses",
            f"landmarks: {len(self.landmarks)}",
        ]
        cv2.rectangle(canvas, (10, 10), (360, 30 + 24 * len(lines)), (18, 22, 28), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                canvas, line, (20, 34 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (235, 240, 245), 1, cv2.LINE_AA,
            )
        cv2.putText(
            canvas, "X -> ; Z up", (20, CANVAS - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 180), 1, cv2.LINE_AA,
        )

        # 右下角相机预览（/image_enhanced），让“地图 + 相机画面”在同一窗口里
        if self.latest_image is not None and self.latest_image.size:
            image = self.latest_image
            height, width = image.shape[:2]
            max_w, max_h = 380, 240
            scale = min(max_w / max(width, 1), max_h / max(height, 1), 1.0)
            thumb = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            thumb_h, thumb_w = thumb.shape[:2]
            x0 = CANVAS - thumb_w - 12
            y0 = CANVAS - thumb_h - 12
            canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
            cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + thumb_w + 2, y0 + thumb_h + 2), (60, 220, 120), 2)
            cv2.putText(
                canvas, f"camera: {self.image_topic}", (x0 + 4, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 245, 250), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas, "no camera image", (CANVAS - 230, CANVAS - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 140, 160), 1, cv2.LINE_AA,
            )
        return canvas

    def on_timer(self) -> None:
        if self._closed:
            return
        try:
            canvas = self._render()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._closed = True
                cv2.destroyWindow(self.window_name)
                self.get_logger().info("可视化窗口已关闭（SLAM 节点仍在运行）")
        except cv2.error as exc:
            self._closed = True
            self.get_logger().error(f"OpenCV 窗口不可用（没有 DISPLAY？）: {exc}")


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = SlamViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
