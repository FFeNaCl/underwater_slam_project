"""SLAM 输入桥接节点：把普通图像话题喂给 ORB-SLAM3 封装包的输入话题。

为什么需要它
------------
``ros2_orb_slam3`` 的 C++ 节点不直接订阅相机话题，而是订阅两个**专用话题**，
并且要先通过 ``/mono_py_driver/experiment_settings`` 把「设置文件名」告诉它，
收到 ``ACK`` 之后才开始吃图：

* ``/mono_py_driver/img_msg``        ``sensor_msgs/Image``
* ``/mono_py_driver/timestep_msg``   ``std_msgs/Float64``（秒，必须严格递增）
* ``/mono_py_driver/experiment_settings`` -> ``/mono_py_driver/exp_settings_ack``（内容 "ACK"）

原仓库自带的 ``mono_driver_node.py`` 只能回放 EuRoC 数据集，本项目需要**实时相机**
和仿真图像，所以这里提供等价但更健壮的桥接：自动握手、时间戳兜底、限频、编码归一化。

参数（``config/slam_bridge.yaml``）
----------------------------------
``image_topic``、``target_image_topic``、``target_timestep_topic``、``config_request_topic``、
``config_ack_topic``、``settings_name``、``use_header_stamp``、``max_rate_hz``、``min_dt_s``、
``handshake_retry_s``、``rehandshake_period_s``、``output_encoding``
"""

from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String

from ..core.params import SlamBridgeConfig, declare_from_flat, load_flat_yaml, resolve_config_path
from .image_bridge import bgr_to_msg, msg_to_bgr, stamp_to_seconds

DEFAULT_CONFIG = "config/slam_bridge.yaml"


class SlamInputBridgeNode(Node):
    """图像 -> ORB-SLAM3 输入话题（含握手与时间戳管理）。"""

    def __init__(self) -> None:
        super().__init__("slam_input_bridge")

        self.declare_parameter("config_file", "")
        config_file = str(self.get_parameter("config_file").value)
        resolved = resolve_config_path(config_file or None, DEFAULT_CONFIG)

        defaults = {
            "image_topic": "/image_enhanced",
            "target_image_topic": "/mono_py_driver/img_msg",
            "target_timestep_topic": "/mono_py_driver/timestep_msg",
            "config_request_topic": "/mono_py_driver/experiment_settings",
            "config_ack_topic": "/mono_py_driver/exp_settings_ack",
            "settings_name": "water_1280x480",
            "use_header_stamp": True,
            "max_rate_hz": 0.0,
            "min_dt_s": 0.001,
            "handshake_retry_s": 0.5,
            "rehandshake_period_s": 0.0,
            "status_topic": "/slam_bridge/status",
            "output_encoding": "mono8",
        }
        if resolved:
            defaults.update({k: v for k, v in load_flat_yaml(resolved).items() if k in defaults})
            self.get_logger().info(f"已加载配置文件: {resolved}")
        declare_from_flat(self, defaults, skip=("config_file",))

        flat = {key: self.get_parameter(key).value for key in defaults}
        self.bridge = SlamBridgeConfig.from_flat(
            {
                "image_topic": flat["image_topic"],
                "target_image_topic": flat["target_image_topic"],
                "target_timestep_topic": flat["target_timestep_topic"],
                "config_request_topic": flat["config_request_topic"],
                "config_ack_topic": flat["config_ack_topic"],
                "settings_name": flat["settings_name"],
                "use_header_stamp": flat["use_header_stamp"],
                "max_rate_hz": flat["max_rate_hz"],
                "min_dt_s": flat["min_dt_s"],
                "handshake_retry_s": flat["handshake_retry_s"],
                "rehandshake_period_s": flat["rehandshake_period_s"],
            }
        )
        self.output_encoding = str(flat["output_encoding"])
        self.status_topic = str(flat["status_topic"])

        self.acked = False
        self.last_handshake = 0.0
        self.last_stamp = 0.0
        self.last_process = 0.0
        self.sent = 0
        self.dropped = 0
        self._min_period = 1.0 / self.bridge.max_rate_hz if self.bridge.max_rate_hz > 0 else 0.0

        self.subscription = self.create_subscription(Image, self.bridge.image_topic, self.on_image, 5)
        self.image_pub = self.create_publisher(Image, self.bridge.target_image_topic, 5)
        self.stamp_pub = self.create_publisher(Float64, self.bridge.target_timestep_topic, 5)
        self.config_pub = self.create_publisher(String, self.bridge.config_request_topic, 5)
        self.status_pub = self.create_publisher(String, self.status_topic, 5)
        self.create_subscription(String, self.bridge.config_ack_topic, self.on_ack, 5)
        self.create_timer(max(0.05, float(self.bridge.handshake_retry_s)), self.on_handshake_timer)

        self.get_logger().info(
            "SLAM 输入桥接启动 | %s -> %s / %s | 设置名 %s | 时间戳来源 %s"
            % (
                self.bridge.image_topic,
                self.bridge.target_image_topic,
                self.bridge.target_timestep_topic,
                self.bridge.settings_name,
                "图像头" if self.bridge.use_header_stamp else "节点时钟",
            )
        )
        self.get_logger().info(
            "注意: 设置文件名会拼成 <settings_file_path_arg>/%s.yaml，"
            "C++ 节点启动时必须把 settings_file_path_arg 指向存放该 yaml 的目录" % self.bridge.settings_name
        )

    # ------------------------------------------------------------------ 握手
    def on_handshake_timer(self) -> None:
        """周期性发送设置名直到收到 ACK（C++ 节点重启后可自动恢复）。"""
        now = time.time()
        period = float(self.bridge.rehandshake_period_s)
        if self.acked and period <= 0.0:
            return
        if self.acked and period > 0.0 and (now - self.last_handshake) < period:
            return
        msg = String()
        msg.data = self.bridge.settings_name
        self.config_pub.publish(msg)
        self.last_handshake = now

    def on_ack(self, msg: String) -> None:
        """收到 ACK 表示 C++ 节点已加载设置文件并初始化 SLAM。"""
        if msg.data.strip().upper() == "ACK":
            if not self.acked:
                self.get_logger().info("收到 ORB-SLAM3 的 ACK，开始转发图像")
            self.acked = True
            self.last_handshake = time.time()

    # ------------------------------------------------------------------ 转发
    def on_image(self, msg: Image) -> None:
        """把图像与时间戳按顺序发布给 ORB-SLAM3 节点。"""
        now = time.time()
        if self._min_period > 0.0 and (now - self.last_process) < self._min_period:
            self.dropped += 1
            return

        # 性能关键：输入已经是目标编码时直接转发，避免 1280x720 下每帧
        # BGR 转换 + 2.7 MB 的 Python 拷贝（实测会把 CPU 打满、帧率拖到 1~3 fps）。
        input_encoding = (msg.encoding or "").lower()
        output_encoding = (self.output_encoding or "bgr8").lower()
        if input_encoding in ("bgr8", "mono8") and input_encoding == output_encoding:
            forwarded = msg
        else:
            try:
                image = msg_to_bgr(msg)
            except Exception as exc:
                self.get_logger().warning(f"图像转换失败: {exc}", throttle_duration_sec=5.0)
                return
            forwarded = bgr_to_msg(
                image, msg.header.stamp, msg.header.frame_id, encoding=output_encoding
            )
        self.last_process = now

        # --- 时间戳：头时间戳优先，但必须严格递增（ORB-SLAM3 依赖它做帧间排序） ---
        stamp: Optional[float] = stamp_to_seconds(msg.header) if self.bridge.use_header_stamp else None
        if stamp is None:
            stamp = now
        if stamp <= self.last_stamp:
            stamp = self.last_stamp + float(self.bridge.min_dt_s)
        self.last_stamp = stamp

        # 关键：把“调整后严格递增”的时间戳写进图像头。
        # C++ 节点现在用图像头作为 SLAM 时间基准，不再依赖另一个 timestep 话题，
        # 避免跨话题到达顺序导致 "timestamp older than previous frame"。
        seconds = int(stamp)
        nanoseconds = int(round((stamp - seconds) * 1e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        forwarded.header.stamp.sec = seconds
        forwarded.header.stamp.nanosec = nanoseconds

        # 仍保留 timestep 话题，兼容旧节点/外部脚本；新 C++ 节点不再依赖它。
        stamp_msg = Float64()
        stamp_msg.data = float(stamp)
        self.stamp_pub.publish(stamp_msg)
        self.image_pub.publish(forwarded)
        self.sent += 1

        if self.sent % 100 == 0:
            status = String()
            status.data = json.dumps(
                {
                    "sent": self.sent,
                    "dropped": self.dropped,
                    "acked": self.acked,
                    "last_stamp": self.last_stamp,
                    "settings_name": self.bridge.settings_name,
                },
                ensure_ascii=True,
            )
            self.status_pub.publish(status)
            self.get_logger().info(f"已转发 {self.sent} 帧，丢弃 {self.dropped} 帧，ACK={self.acked}")


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = SlamInputBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
