"""图像序列 / 合成场景播放节点：**没有相机也能把整条链路跑起来**。

这是本项目的验证基础设施：Linux 侧 AI（或任何拿不到相机的人）可以用它
在几秒钟内验证「增强 -> 识别 -> 桥接 -> SLAM -> 语义地标 -> RViz」整条链路，
并把问题定位到具体某一环。

两种数据源
----------
1. ``synthetic: true``（默认）：用 ``core.synthetic`` 实时渲染水下红色立方体场景，
   目标会缓慢移动，因此跟踪、三角化都能被驱动；**不需要任何数据文件**。
2. ``image_dir: /path/to/images``：按文件名排序回放真实图片（可用 ``v4l2_camera`` 抓的图、
   或 EuRoC/VRX 的截图），用于真实数据回归。

参数
----
``topic``（默认 ``/image_raw``）、``fps``（默认 10）、``loop``、``frame_id``、
``synthetic``、``width``/``height``、``cube_edge_m``、``distance_m``、``lateral_speed_mps``、
``num_frames``（0 表示无限循环）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ..core.params import declare_from_flat
from ..core.synthetic import default_camera_matrix, render_cube_scene, render_underwater_background
from .image_bridge import bgr_to_msg

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class ImageSequencePlayerNode(Node):
    """按固定帧率发布图像（合成或磁盘序列）。"""

    def __init__(self) -> None:
        super().__init__("image_sequence_player")

        defaults = {
            "topic": "/image_raw",
            "image_dir": "",
            "fps": 10.0,
            "loop": True,
            "frame_id": "camera_optical",
            "synthetic": True,
            "width": 1280,
            "height": 480,
            "cube_edge_m": 0.05,
            "distance_m": 0.45,
            "lateral_speed_mps": 0.06,
            "num_frames": 0,
            "start_index": 0,
        }
        declare_from_flat(self, defaults)
        flat = {key: self.get_parameter(key).value for key in defaults}

        self.topic = str(flat["topic"])
        self.image_dir = str(flat["image_dir"])
        self.fps = max(0.1, float(flat["fps"]))
        self.loop = bool(flat["loop"])
        self.frame_id = str(flat["frame_id"])
        self.synthetic = bool(flat["synthetic"])
        self.width = int(flat["width"])
        self.height = int(flat["height"])
        self.cube_edge_m = float(flat["cube_edge_m"])
        self.distance_m = float(flat["distance_m"])
        self.lateral_speed = float(flat["lateral_speed_mps"])
        self.num_frames = int(flat["num_frames"])
        self.start_index = int(flat["start_index"])

        self.publisher = self.create_publisher(Image, self.topic, 5)
        self.files: List[Path] = []
        self.index = 0
        self.published = 0
        self._background: Optional[np.ndarray] = None
        self._camera = default_camera_matrix(self.width, self.height)
        self._finished = False

        if not self.synthetic and self.image_dir:
            directory = Path(self.image_dir).expanduser()
            if directory.is_dir():
                self.files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                self.get_logger().info(f"回放目录 {directory}，共 {len(self.files)} 张图")
            else:
                self.get_logger().error(f"image_dir 不存在: {directory}，回退到合成模式")
                self.synthetic = True
        if self.synthetic:
            self._background = render_underwater_background(self.width, self.height, seed=42)
            self.get_logger().info(
                f"合成模式: 实时渲染水下红色立方体（边长 {self.cube_edge_m} m，距离约 {self.distance_m} m）"
            )

        self.create_timer(1.0 / self.fps, self.on_timer)
        self.get_logger().info(f"图像播放节点启动 | 话题 {self.topic} | {self.fps:.1f} FPS | frame_id={self.frame_id}")

    def on_timer(self) -> None:
        """发布下一帧；序列回放结束后按 ``loop`` 决定是否循环。"""
        if self._finished:
            return
        image, stamp = self._next_frame()
        if image is None:
            return
        self.publisher.publish(bgr_to_msg(image, stamp, self.frame_id))
        self.published += 1
        if self.published % 50 == 0:
            self.get_logger().info(f"已发布 {self.published} 帧")

    def _next_frame(self):
        """取下一帧图像与其时间戳（秒）。"""
        stamp = self.get_clock().now().to_msg()
        if self.synthetic:
            if self.num_frames > 0 and self.index >= self.num_frames:
                self._finished = True
                self.get_logger().info(f"合成序列播放完成（{self.index} 帧）")
                return None, stamp
            # 目标沿 x 方向缓慢往返移动（三角波），保证每帧形状/位置都有变化，能驱动跟踪与三角化
            phase = (self.index % 200) / 200.0
            offset = (2.0 * abs(2.0 * phase - 1.0) - 1.0) * self.lateral_speed * 2.0
            rvec = (0.35 + 0.15 * phase, 0.45 - 0.1 * phase, 0.15)
            tvec = (offset, 0.0, self.distance_m)
            scene = render_cube_scene(
                self._camera,
                rvec=rvec,
                tvec=tvec,
                edge_m=self.cube_edge_m,
                image_size=(self.width, self.height),
                background=self._background,
                seed=self.index % 7,
            )
            self.index += 1
            return scene.image, stamp

        if not self.files:
            self._finished = True
            self.get_logger().error("没有可回放的图像文件")
            return None, stamp
        if self.index >= len(self.files):
            if not self.loop:
                self._finished = True
                self.get_logger().info(f"序列播放完成（{len(self.files)} 张）")
                return None, stamp
            self.index = 0
        path = self.files[self.index]
        # 用 imdecode 读，兼容中文/空格路径
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        self.index += 1
        if image is None:
            self.get_logger().warning(f"读取失败: {path}", throttle_duration_sec=5.0)
            return None, stamp
        return image, stamp


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = ImageSequencePlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
