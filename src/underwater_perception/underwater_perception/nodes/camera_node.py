"""轻量 USB 相机节点：OpenCV 取流 -> ``/image_raw``。

为什么需要它：本机实测 ``v4l2_camera`` 在 640x480 YUYV 下只有约 10 fps
（内部 yuv422->rgb8 转换慢），``usb_cam`` 的 mjpeg2rgb 约 7 fps；
而同一台相机用 OpenCV 直接读 MJPG 1280x720 可到约 22 fps、640x480 约 25~28 fps。
本节点只做“取帧 + 发布”，不做额外转换，适合作为真实相机链路的入口。

用法::

    ros2 run underwater_perception camera --ros-args \
      -p camera_index:=0 -p width:=1280 -p height:=720 \
      -p fourcc:=MJPG -p fps:=30.0 -p topic:=/image_raw

注意：``camera_index`` 对应 ``/dev/video0`` -> 0；``/dev/video1``、``/dev/video3``
通常是 UVC metadata 节点，打不开。
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ..core.camera_io import DEFAULT_FOURCC, DEFAULT_FPS, camera_info, open_camera
from .image_bridge import bgr_to_msg


class CameraNode(Node):
    """OpenCV 取流并发布 ``sensor_msgs/Image``（bgr8）。"""

    def __init__(self) -> None:
        super().__init__("camera")

        self.camera_index = int(self.declare_parameter("camera_index", 0).value)
        self.device = str(self.declare_parameter("device", "").value)
        self.width = int(self.declare_parameter("width", 1280).value)
        self.height = int(self.declare_parameter("height", 720).value)
        self.fourcc = str(self.declare_parameter("fourcc", DEFAULT_FOURCC).value)
        self.fps = float(self.declare_parameter("fps", DEFAULT_FPS).value)
        self.topic = str(self.declare_parameter("topic", "/image_raw").value)
        self.frame_id = str(self.declare_parameter("frame_id", "camera_optical").value)
        self.log_interval_s = float(self.declare_parameter("log_interval_s", 5.0).value)
        self.reopen_interval_s = float(self.declare_parameter("reopen_interval_s", 2.0).value)

        self.source = self.device or self.camera_index
        self.capture = open_camera(
            self.source,
            width=self.width,
            height=self.height,
            fourcc=self.fourcc,
            fps=self.fps,
        )
        if not self.capture.isOpened():
            self.get_logger().error(
                f"无法打开相机 source={self.source!r}。"
                " 可用 v4l2-ctl --list-devices 查看当前节点；"
                " UVC 重新枚举后序号会变化，建议传 device:=/dev/v4l/by-id/...。"
            )
            raise SystemExit(1)

        info = camera_info(self.capture)
        self.get_logger().info(
            "相机已打开: {fourcc} {width}x{height} @ {fps:.1f} fps -> {topic} (frame_id={frame_id})".format(
                topic=self.topic, frame_id=self.frame_id, **info
            )
        )
        if info.get("fourcc") != "MJPG" and info.get("width", 0) >= 1280:
            self.get_logger().warning(
                "当前不是 MJPG，1280x720 的 YUYV 在 USB2 上通常只有 5~10 fps"
            )

        self.publisher = self.create_publisher(Image, self.topic, 10)
        self.timer = self.create_timer(1.0 / max(self.fps, 1.0), self.on_timer)
        self.frames = 0
        self.failures = 0
        self.reopen_count = 0
        self._last_reopen = 0.0
        self._last_log = time.time()
        self._last_frames = 0
        self._last_time = time.time()

    def _reopen(self) -> None:
        """UVC 重新枚举/短暂掉线后，释放旧句柄并按原 source 重开。

        实测：两套 pipeline 同时抢同一个 Lightburn 相机时，内核会把设备从
        video2 重枚举到 video3；旧 VideoCapture 句柄随即失效，read() 永远失败。
        仅计数不重开会让整个链路看起来“卡死”。
        """
        try:
            self.capture.release()
        except Exception:
            pass
        self.capture = open_camera(
            self.source,
            width=self.width,
            height=self.height,
            fourcc=self.fourcc,
            fps=self.fps,
        )
        if self.capture.isOpened():
            self.reopen_count += 1
            self.failures = 0
            self.get_logger().info(f"相机重连成功（第 {self.reopen_count} 次）: {self.source!r}")
        else:
            self.get_logger().error(
                f"相机重连失败: {self.source!r}；请确认设备仍在、且没有第二个节点占用",
                throttle_duration_sec=10.0,
            )

    def on_timer(self) -> None:
        """读一帧并发布；连续读失败时尝试重连，不退出节点。"""
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.failures += 1
            now = time.time()
            if (now - self._last_reopen) >= self.reopen_interval_s:
                self._last_reopen = now
                self.get_logger().warning(
                    f"读帧失败（累计 {self.failures} 次），尝试重连相机"
                )
                self._reopen()
            return

        self.publisher.publish(
            bgr_to_msg(frame, self.get_clock().now().to_msg(), self.frame_id)
        )
        self.frames += 1

        now = time.time()
        if self.log_interval_s > 0.0 and (now - self._last_log) >= self.log_interval_s:
            elapsed = max(now - self._last_time, 1e-6)
            measured = (self.frames - self._last_frames) / elapsed
            self.get_logger().info(
                f"发布 {self.frames} 帧，实测 {measured:.1f} fps，失败 {self.failures} 次"
            )
            self._last_log = now
            self._last_frames = self.frames
            self._last_time = now

    def destroy_node(self) -> None:
        """释放相机。"""
        try:
            self.capture.release()
        finally:
            super().destroy_node()


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
