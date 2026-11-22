"""语义地标节点：把检测结果与 SLAM 位姿三角化成地图中的三维目标。

输入
----
``detections_topic``  默认 ``/detections``（``std_msgs/String`` JSON，几何旁路或 YOLO 节点均可）
``pose_topic``        默认 ``/slam/pose``（``geometry_msgs/PoseStamped``，**Twc**：相机在世界系）
``status_topic``      默认 ``/slam/status``（``std_msgs/String``，可选，用于判断跟踪是否正常）

输出
----
``markers_topic``     默认 ``/semantic/markers``（``visualization_msgs/MarkerArray``）
``landmarks_topic``   默认 ``/semantic/landmarks``（``std_msgs/String`` JSON）

两种工作模式（消息里 ``source`` 字段会明确标注，避免误读）
--------------------------------------------------------
``triangulated``    有 SLAM 位姿：多视图射线三角化，与点云/轨迹**同一尺度**，可信
``scale_fallback``  无 SLAM 位姿：用已知立方体边长估计距离，得到**相机坐标系**下的近似位置，
                    仅用于"SLAM 还没起来时屏幕上也能看到目标"，不代表地图尺度

参数见 ``config/semantic_landmark.yaml``。相机内参用于由像素反算射线；
若检测消息里已经带了 ``measurement.ray_camera``（几何旁路默认会带），则内参可省略。
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ..core.landmarks import Landmark, LandmarkManager
from ..core.measurement import pose_to_matrix
from ..core.params import LandmarkConfig, declare_from_flat, load_flat_yaml, resolve_config_path
from ..core.types import DetectionFrame

DEFAULT_CONFIG = "config/semantic_landmark.yaml"


class SemanticLandmarkNode(Node):
    """维护语义地标并发布 MarkerArray / JSON。"""

    def __init__(self) -> None:
        super().__init__("semantic_landmark")

        self.declare_parameter("config_file", "")
        config_file = str(self.get_parameter("config_file").value)
        resolved = resolve_config_path(config_file or None, DEFAULT_CONFIG)

        defaults = {
            "detections_topic": "/detections",
            "pose_topic": "/slam/pose",
            "status_topic": "/slam/status",
            "landmarks_topic": "/semantic/landmarks",
            "markers_topic": "/semantic/markers",
            "min_views": 2,
            "min_parallax_deg": 2.0,
            "min_baseline_m": 0.02,
            "max_reproj_error_px": 12.0,
            "max_observations": 30,
            "stale_after_s": 5.0,
            "allow_scale_fallback": True,
            "cube_edge_m": 0.05,
            "landmark_class_names": [],
            "min_detection_confidence": 0.0,
            "publish_period_s": 0.5,
            "max_pose_age_s": 1.0,
            "camera_fx": 0.0,
            "camera_fy": 0.0,
            "camera_cx": 0.0,
            "camera_cy": 0.0,
        }
        if resolved:
            defaults.update({k: v for k, v in load_flat_yaml(resolved).items() if k in defaults})
            self.get_logger().info(f"已加载配置文件: {resolved}")
        declare_from_flat(self, defaults, skip=("config_file",))

        flat = {key: self.get_parameter(key).value for key in defaults}
        self.config = LandmarkConfig.from_flat(flat)
        self.publish_period_s = float(flat["publish_period_s"])
        self.max_pose_age_s = float(flat["max_pose_age_s"])
        self.manager = LandmarkManager(self.config)

        self.camera_matrix = None
        if all(float(flat[key]) > 0.0 for key in ("camera_fx", "camera_fy", "camera_cx", "camera_cy")):
            self.camera_matrix = np.array(
                [
                    [float(flat["camera_fx"]), 0.0, float(flat["camera_cx"])],
                    [0.0, float(flat["camera_fy"]), float(flat["camera_cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        self._latest_pose: Optional[np.ndarray] = None
        self._latest_pose_stamp = 0.0
        self._latest_pose_frame = "map"
        self._landmarks_seen = 0

        self.create_subscription(String, self.config.detections_topic, self.on_detections, 10)
        self.create_subscription(PoseStamped, self.config.pose_topic, self.on_pose, 10)
        self.create_subscription(String, self.config.status_topic, self.on_status, 10)
        self.markers_pub = self.create_publisher(MarkerArray, self.config.markers_topic, 10)
        self.landmarks_pub = self.create_publisher(String, self.config.landmarks_topic, 10)
        self.create_timer(max(0.05, self.publish_period_s), self.on_timer)

        self.get_logger().info(
            "语义地标节点启动 | 检测 %s + 位姿 %s -> 地标 %s / Marker %s | 最少视图 %d 视差 >= %.1f°"
            % (
                self.config.detections_topic,
                self.config.pose_topic,
                self.config.landmarks_topic,
                self.config.markers_topic,
                self.config.min_views,
                self.config.min_parallax_deg,
            )
        )
        if self.camera_matrix is None:
            self.get_logger().info("未配置相机内参：将只使用检测消息里自带的 measurement.ray_camera 反算射线")
        if self.config.landmark_class_names:
            self.get_logger().info(
                "地标类别白名单: %s（其他类别会被忽略，避免动态目标进地图）"
                % ", ".join(self.config.landmark_class_names)
            )
        if self.config.min_detection_confidence > 0.0:
            self.get_logger().info(
                "地标检测置信度下限: %.2f" % self.config.min_detection_confidence
            )

    # ------------------------------------------------------------------ 回调
    def on_pose(self, msg: PoseStamped) -> None:
        """记录最新相机位姿（Twc）。"""
        pose = msg.pose
        self._latest_pose = pose_to_matrix(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        )
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._latest_pose_stamp = stamp if stamp > 0.0 else time.time()
        self._latest_pose_frame = msg.header.frame_id or "map"
        self.manager.set_last_pose(self._latest_pose, self._latest_pose_stamp)

    def on_status(self, msg: String) -> None:
        """跟踪状态异常时打一次日志（LOST 期间三角化结果不可信）。"""
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        state = str(payload.get("tracking_state", ""))
        if state in ("LOST", "RECENTLY_LOST", "NOT_INITIALIZED") and not getattr(self, "_warned_state", False):
            self.get_logger().warning(f"SLAM 跟踪状态为 {state}，此时不会产生新的三角化地标")
            self._warned_state = True
        elif state in ("OK", "OK_KLT"):
            self._warned_state = False

    def on_detections(self, msg: String) -> None:
        """把一帧检测交给地标管理器。"""
        try:
            frame = DetectionFrame.from_json(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"检测 JSON 解析失败: {exc}", throttle_duration_sec=5.0)
            return

        pose = None
        frame_id = frame.frame_id or self._latest_pose_frame
        if self._latest_pose is not None:
            age = time.time() - self._latest_pose_stamp
            if age <= self.max_pose_age_s:
                pose = self._latest_pose
                frame_id = self._latest_pose_frame or frame_id

        for detection in frame.detections:
            if not self.config.accepts(detection.class_name, detection.confidence):
                continue
            landmark = self.manager.add_detection(
                detection,
                stamp=float(frame.stamp_sec) + float(frame.stamp_nanosec) * 1e-9,
                pose_wc=pose,
                frame_id=frame_id,
                camera_matrix=self.camera_matrix,
            )
            if landmark is not None:
                self._landmarks_seen += 1

    def on_timer(self) -> None:
        """周期性发布地标（Marker + JSON）。"""
        self.manager.prune()
        landmarks = self.manager.snapshot()
        if not landmarks:
            return
        self.markers_pub.publish(self._build_markers(landmarks))
        payload = String()
        payload.data = json.dumps(
            {
                "count": len(landmarks),
                "landmarks": [landmark.to_dict() for landmark in landmarks],
                "note": "triangulated 与地图同尺度；scale_fallback 为相机系近似，不代表地图尺度",
            },
            ensure_ascii=True,
        )
        self.landmarks_pub.publish(payload)

    # ------------------------------------------------------------------ Marker
    def _build_markers(self, landmarks) -> MarkerArray:
        """三角化地标画绿色立方体，fallback 地标画黄色（颜色本身就是可信度提示）。"""
        array = MarkerArray()
        edge = float(self.config.cube_edge_m)
        for index, landmark in enumerate(landmarks):
            color = (0.2, 0.9, 0.35) if landmark.source == "triangulated" else (0.95, 0.8, 0.2)

            cube = Marker()
            cube.header.frame_id = landmark.frame_id or "map"
            cube.header.stamp = self.get_clock().now().to_msg()
            cube.ns = "semantic_landmarks"
            cube.id = int(landmark.landmark_id)
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.scale.x = cube.scale.y = cube.scale.z = max(edge, 1e-3)
            cube.pose.position.x = float(landmark.position[0])
            cube.pose.position.y = float(landmark.position[1])
            cube.pose.position.z = float(landmark.position[2])
            cube.pose.orientation.w = 1.0
            cube.color.a = 0.7
            cube.color.r, cube.color.g, cube.color.b = color
            array.markers.append(cube)

            text = Marker()
            text.header = cube.header
            text.ns = "semantic_labels"
            text.id = 10000 + int(landmark.landmark_id)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = cube.pose
            text.pose.position.z -= edge * 1.2
            text.scale.z = max(0.03, edge * 0.8)
            label = f"L{landmark.landmark_id}/{landmark.class_name} v{landmark.n_views} {landmark.source}"
            if landmark.distance_m is not None:
                label += f" {landmark.distance_m:.2f}m"
            text.text = label
            text.color.a = 1.0
            text.color.r = text.color.g = text.color.b = 1.0
            array.markers.append(text)
        return array


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = SemanticLandmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
