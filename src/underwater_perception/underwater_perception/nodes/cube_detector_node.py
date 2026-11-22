"""几何旁路立方体识别节点（ROS2 薄封装）。

职责边界：本文件只做「参数声明 / 消息转换 / 发布」，所有算法都在
``underwater_perception.core`` 里（那些代码可以在没有 ROS 的机器上单测）。
若发现本文件里出现了几何或数学逻辑，说明分层被破坏了。

话题契约
--------
订阅:
    ``input_topic``                  默认 ``/image_enhanced``（``sensor_msgs/Image``）
发布:
    ``detections_topic``             默认 ``/detections``（``std_msgs/String``，JSON）
        —— 与既有 ``yolo_vrx_bridge`` 的输出**同 schema 且为其超集**，
        详见 ``underwater_perception/core/types.py`` 与 ``docs/INTERFACES.md``
    ``annotated_topic``              默认 ``/detections/annotated``（``sensor_msgs/Image``）
    ``markers_topic``                默认 ``/detections/markers``（``visualization_msgs/MarkerArray``）

参数：全部通过 ``config/cube_detector.yaml``（ROS 参数文件）配置，键名与
``CubeDetectorConfig.to_flat()`` 一一对应，支持 ``ros2 run ... --ros-args -p key:=value`` 覆盖。

使用示例
--------
.. code-block:: bash

    ros2 run underwater_perception cube_detector --ros-args \\
      --params-file $(ros2 pkg prefix underwater_perception)/share/underwater_perception/config/cube_detector.yaml
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ..core.cube_geometry import CubeDetector
from ..core.params import CubeDetectorConfig, declare_from_flat, dump_flat_yaml, load_flat_yaml, resolve_config_path
from ..core.types import DetectionFrame
from ..core.visualize import annotate_detections
from .image_bridge import bgr_to_msg, msg_to_bgr, stamp_to_seconds

DEFAULT_CONFIG = "config/cube_detector.yaml"


class CubeDetectorNode(Node):
    """把 ``core.cube_geometry.CubeDetector`` 接到 ROS 话题上。"""

    def __init__(self) -> None:
        super().__init__("cube_detector")

        # --- 1) 先确定配置文件，再用「文件值 -> 覆盖 dataclass 默认值」的顺序声明参数 ---
        # 这样 ``ros2 run ... -p key:=value`` 仍能覆盖文件值（命令行参数优先级最高）
        self.declare_parameter("config_file", "")
        config_file = str(self.get_parameter("config_file").value)
        resolved = resolve_config_path(config_file or None, DEFAULT_CONFIG)
        flat_defaults = CubeDetectorConfig().to_flat()
        if resolved:
            file_values = load_flat_yaml(resolved)
            flat_defaults.update({k: v for k, v in file_values.items() if k in flat_defaults})
            self.get_logger().info(f"已加载配置文件: {resolved}")
        else:
            self.get_logger().warning(
                f"未找到配置文件（config_file='{config_file}'，默认 '{DEFAULT_CONFIG}'），使用内置默认值"
            )

        declare_from_flat(self, flat_defaults, skip=("config_file",))
        flat = {key: self.get_parameter(key).value for key in flat_defaults}
        self.config = CubeDetectorConfig.from_flat(flat)
        self.detector = CubeDetector(self.config)

        # --- 2) 参数缓存（避免每帧访问 ROS 参数） ---
        self.input_topic = self.config.input_topic
        self.detections_topic = self.config.detections_topic
        self.annotated_topic = self.config.annotated_topic
        self.markers_topic = self.config.markers_topic
        self.max_rate_hz = float(self.config.max_rate_hz)
        self._min_period = 1.0 / self.max_rate_hz if self.max_rate_hz > 0 else 0.0
        self._last_process_time = 0.0
        self._frame_count = 0
        self._processed_count = 0
        self._detection_count = 0
        self._fps_window_start = time.time()
        self._fps = 0.0

        # --- 3) 话题 ---
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(Image, self.input_topic, self.on_image, qos)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        self.annotated_pub = (
            self.create_publisher(Image, self.annotated_topic, 10) if self.config.publish_annotated else None
        )
        self.markers_pub = (
            self.create_publisher(MarkerArray, self.markers_topic, 10) if self.config.publish_markers else None
        )

        self.get_logger().info(
            "几何旁路立方体识别已启动 | 输入 %s -> 检测 %s | 颜色预设=%s 模式=%s 最小置信度=%.2f 立方体边长=%.3f m"
            % (
                self.input_topic,
                self.detections_topic,
                self.config.color.preset,
                self.config.detect.mode,
                self.config.detect.min_confidence,
                self.config.measure.cube_edge_m,
            )
        )
        self.get_logger().info(f"生效参数（扁平，可用于复现）:\n{dump_flat_yaml(self.config.to_flat(), 'cube_detector')}")

    # ------------------------------------------------------------------ 主回调
    def on_image(self, msg: Image) -> None:
        """每帧：检测 -> 发布标注图/检测 JSON/可视化 Marker。"""
        now = time.time()
        self._frame_count += 1
        if self._min_period > 0.0 and (now - self._last_process_time) < self._min_period:
            return
        self._last_process_time = now

        try:
            image = msg_to_bgr(msg)
        except Exception as exc:  # 编码不支持/数据异常时只记一次警告，避免刷屏
            self.get_logger().warning(f"图像转换失败: {exc}", throttle_duration_sec=5.0)
            return

        stamp = stamp_to_seconds(msg.header)
        detections = self.detector.detect(image, stamp=stamp)
        self._processed_count += 1
        self._detection_count += len(detections)

        # --- 检测结果 JSON（与 yolo 节点兼容的超集） ---
        frame = DetectionFrame(
            detections=detections,
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
            frame_id=msg.header.frame_id,
            image_width=int(msg.width),
            image_height=int(msg.height),
            source="geometry",
            meta={"fps": round(self._fps, 2), "detect_width": self.config.detect.detect_width},
        )
        out = String()
        out.data = frame.to_json()
        self.detections_pub.publish(out)

        # --- 标注图（可直接 rqt_image_view 查看） ---
        if self.annotated_pub is not None:
            title = (
                f"geometry cube detector | {len(detections)} obj | {self._fps:.1f} FPS | "
                f"mode={self.config.detect.mode}"
            )
            annotated = annotate_detections(
                image, detections, config=self.config, mask=None, title=title, show_metrics=True
            )
            self.annotated_pub.publish(bgr_to_msg(annotated, msg.header.stamp, msg.header.frame_id))

        # --- 可视化 Marker（放在相机坐标系，便于 RViz 里直接看到方向与距离） ---
        if self.markers_pub is not None and detections:
            self.markers_pub.publish(self._build_markers(detections, msg.header))

        # --- 统计 ---
        elapsed = now - self._fps_window_start
        if elapsed >= 2.0:
            self._fps = self._processed_count / elapsed
            self._processed_count = 0
            self._fps_window_start = now
            if self._frame_count % 60 == 0:
                self.get_logger().info(
                    f"几何识别统计: 累计帧 {self._frame_count}, 累计检测 {self._detection_count}, 当前 {self._fps:.1f} FPS"
                )

    # ------------------------------------------------------------------ Marker
    def _build_markers(self, detections, header) -> MarkerArray:
        """把检测结果画成 RViz Marker：立方体（有距离时放在真实位置）+ 文字标签。"""
        array = MarkerArray()
        edge = float(self.config.measure.cube_edge_m)
        for index, detection in enumerate(detections):
            measurement = detection.measurement
            distance: Optional[float] = None
            ray = None
            if measurement is not None:
                distance = measurement.distance_m
                ray = measurement.ray_camera

            cube = Marker()
            cube.header = header
            cube.ns = "cube_detections"
            cube.id = index
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.scale.x = cube.scale.y = cube.scale.z = max(edge, 1e-3)
            if ray is not None and distance is not None:
                cube.pose.position.x = float(ray[0]) * distance
                cube.pose.position.y = float(ray[1]) * distance
                cube.pose.position.z = float(ray[2]) * distance
                # 用拟合出的立方体姿态（若可得）让方块朝向更接近真实
                if measurement is not None and measurement.pose_rvec is not None:
                    self._set_orientation_from_rvec(cube, measurement.pose_rvec)
            else:
                # 没有距离时按固定 1 m 摆在光轴方向上，仅用于提示"检到了"，不代表真实位置
                cube.pose.position.x = 0.0
                cube.pose.position.y = 0.0
                cube.pose.position.z = 1.0
            cube.color.a = 0.55
            cube.color.r = 0.9
            cube.color.g = 0.25
            cube.color.b = 0.2
            array.markers.append(cube)

            text = Marker()
            text.header = header
            text.ns = "cube_labels"
            text.id = 1000 + index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.scale.z = max(0.03, edge * 0.8)
            text.pose = cube.pose
            text.pose.position.z = cube.pose.position.z - edge * 0.8
            label = f"#{detection.track_id} {detection.confidence:.2f}"
            if distance is not None:
                label += f" {distance:.2f}m"
            text.text = label
            text.color.a = 1.0
            text.color.r = text.color.g = text.color.b = 1.0
            array.markers.append(text)
        return array

    @staticmethod
    def _set_orientation_from_rvec(marker: Marker, rvec) -> None:
        """Rodrigues 向量 -> 四元数（把立方体姿态填进 Marker）。"""
        import numpy as np

        try:
            import cv2

            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        except Exception:
            return
        trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w, x = 0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s
            y, z = (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        marker.pose.orientation.x = float(x)
        marker.pose.orientation.y = float(y)
        marker.pose.orientation.z = float(z)
        marker.pose.orientation.w = float(w)


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = CubeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
