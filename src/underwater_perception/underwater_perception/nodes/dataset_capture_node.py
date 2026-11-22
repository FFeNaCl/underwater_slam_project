"""真实相机数据集采集节点（ROS2）。

订阅相机/增强图像话题，按固定间隔把帧保存到 ``<output_dir>/images``，
供后续 YOLO 标注与训练。与 ``tools/capture_dataset.py`` 使用同一套落盘逻辑，
区别是这里直接吃 ROS 话题，适合真实相机已经通过 ``v4l2_camera`` 发布的场景。

服务
----
``~/start``     (``std_srvs/Trigger``)  开始/继续保存
``~/stop``      (``std_srvs/Trigger``)  暂停保存
``~/snapshot``  (``std_srvs/Trigger``)  立即保存最近一帧
``~/reset``     (``std_srvs/Trigger``)  计数归零（不删除已保存文件）

话题
----
订阅 ``image_topic``（默认 ``/image_enhanced``）
发布 ``status_topic``（默认 ``/dataset_capture/status``，JSON）
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ..core.dataset_io import (
    ensure_dataset_dirs,
    frame_filename,
    imwrite_unicode,
    write_session_json,
)
from ..core.params import declare_from_flat, load_flat_yaml, resolve_config_path
from .image_bridge import msg_to_bgr, stamp_to_seconds

DEFAULT_CONFIG = "config/dataset_capture.yaml"


class DatasetCaptureNode(Node):
    """把 ROS 图像话题按间隔落盘为 YOLO 数据集原始帧。"""

    def __init__(self) -> None:
        super().__init__("dataset_capture")

        self.declare_parameter("config_file", "")
        config_file = str(self.get_parameter("config_file").value)
        resolved = resolve_config_path(config_file or None, DEFAULT_CONFIG)

        defaults = {
            "image_topic": "/image_enhanced",
            "output_dir": "~/underwater_slam_project/datasets/real_underwater/raw",
            "save_interval_s": 0.5,
            "max_images": 0,
            "prefix": "frame",
            "image_format": "jpg",
            "write_empty_labels": False,
            "enabled": True,
            "min_width": 0,
            "min_height": 0,
            "status_topic": "/dataset_capture/status",
        }
        if resolved:
            defaults.update({k: v for k, v in load_flat_yaml(resolved).items() if k in defaults})
            self.get_logger().info(f"已加载配置文件: {resolved}")
        declare_from_flat(self, defaults, skip=("config_file",))

        flat = {key: self.get_parameter(key).value for key in defaults}
        self.image_topic = str(flat["image_topic"])
        self.output_dir = str(Path(str(flat["output_dir"])).expanduser())
        self.save_interval_s = float(flat["save_interval_s"])
        self.max_images = int(flat["max_images"])
        self.prefix = str(flat["prefix"])
        self.image_format = str(flat["image_format"])
        self.write_empty_labels = bool(flat["write_empty_labels"])
        self.enabled = bool(flat["enabled"])
        self.min_width = int(flat["min_width"])
        self.min_height = int(flat["min_height"])
        self.status_topic = str(flat["status_topic"])

        self.images_dir, self.labels_dir = ensure_dataset_dirs(self.output_dir)
        self.saved = 0
        self.skipped = 0
        self.failed = 0
        self.last_save = 0.0
        self.last_file = ""
        self._last_image = None
        self._last_stamp_s = 0.0
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        self.create_subscription(Image, self.image_topic, self.on_image, 5)
        self.status_pub = self.create_publisher(String, self.status_topic, 5)
        self.create_service(Trigger, "~/start", self.on_start)
        self.create_service(Trigger, "~/stop", self.on_stop)
        self.create_service(Trigger, "~/snapshot", self.on_snapshot)
        self.create_service(Trigger, "~/reset", self.on_reset)

        self.get_logger().info(
            "数据集采集节点启动 | 输入 %s | 输出 %s | 间隔 %.2f s | 上限 %s"
            % (
                self.image_topic,
                self.images_dir,
                self.save_interval_s,
                self.max_images if self.max_images > 0 else "不限",
            )
        )
        self.get_logger().info(
            "服务: ~/start ~/stop ~/snapshot ~/reset；"
            "建议用同一段连续视频抽帧后按场次拆分 train/valid，避免相邻帧泄漏。"
        )

    # ------------------------------------------------------------------ 工具
    def _save_image(self, image, stamp_s: float) -> bool:
        """保存一帧并更新计数；返回是否成功。"""
        if self.max_images > 0 and self.saved >= self.max_images:
            return False
        if image is None:
            return False
        if self.min_width > 0 and image.shape[1] < self.min_width:
            self.skipped += 1
            return False
        if self.min_height > 0 and image.shape[0] < self.min_height:
            self.skipped += 1
            return False

        stamp_ns = int(stamp_s * 1e9) if stamp_s > 0.0 else int(time.time() * 1e9)
        filename = frame_filename(self.prefix, self.saved, stamp_ns, self.image_format)
        target = self.images_dir / filename
        if not imwrite_unicode(target, image):
            self.failed += 1
            self.get_logger().error(f"写入失败: {target}")
            return False
        if self.write_empty_labels:
            (self.labels_dir / f"{Path(filename).stem}.txt").write_text("", encoding="utf-8")
        self.saved += 1
        self.last_save = time.time()
        self.last_file = str(target)
        self.get_logger().info(f"保存 {self.saved}: {target}")
        if self.saved % 10 == 0:
            self._write_session()
        self._publish_status()
        return True

    def _write_session(self) -> None:
        payload = {
            "started_at": self._started_at,
            "source": "ros:" + self.image_topic,
            "image_dir": str(self.images_dir),
            "label_dir": str(self.labels_dir),
            "saved": int(self.saved),
            "skipped": int(self.skipped),
            "failed": int(self.failed),
            "interval_s": float(self.save_interval_s),
            "image_format": self.image_format,
            "last_file": self.last_file,
            "note": "训练/验证集按视频或场次拆分，不要随机拆相邻帧。",
        }
        try:
            write_session_json(self.output_dir, payload)
        except OSError as exc:
            self.get_logger().warning(f"写会话记录失败: {exc}")

    def _publish_status(self) -> None:
        status = String()
        status.data = json.dumps(
            {
                "enabled": bool(self.enabled),
                "saved": int(self.saved),
                "skipped": int(self.skipped),
                "failed": int(self.failed),
                "max_images": int(self.max_images),
                "output_dir": str(self.images_dir),
                "last_file": self.last_file,
            },
            ensure_ascii=True,
        )
        self.status_pub.publish(status)

    # ------------------------------------------------------------------ 回调
    def on_image(self, msg: Image) -> None:
        """按间隔保存图像。"""
        if not self.enabled:
            return
        if self.max_images > 0 and self.saved >= self.max_images:
            return

        try:
            image = msg_to_bgr(msg)
        except Exception as exc:
            self.failed += 1
            self.get_logger().warning(f"图像转换失败: {exc}", throttle_duration_sec=5.0)
            return

        stamp = stamp_to_seconds(msg.header)
        self._last_image = image
        self._last_stamp_s = float(stamp if stamp is not None else time.time())

        now = time.time()
        if (now - self.last_save) < self.save_interval_s:
            self.skipped += 1
            return
        self._save_image(image, self._last_stamp_s)

    def on_start(self, request, response):
        self.enabled = True
        response.success = True
        response.message = f"开始保存到 {self.images_dir}"
        self.get_logger().info(response.message)
        self._publish_status()
        return response

    def on_stop(self, request, response):
        self.enabled = False
        self._write_session()
        response.success = True
        response.message = f"已暂停保存（累计 {self.saved} 张）"
        self.get_logger().info(response.message)
        self._publish_status()
        return response

    def on_snapshot(self, request, response):
        if self._last_image is None:
            response.success = False
            response.message = "尚未收到任何图像"
            return response
        ok = self._save_image(self._last_image, self._last_stamp_s)
        response.success = bool(ok)
        response.message = f"已保存: {self.last_file}" if ok else "保存失败（可能已达上限或写盘错误）"
        return response

    def on_reset(self, request, response):
        previous = self.saved
        self.saved = 0
        self.skipped = 0
        self.failed = 0
        self.last_save = 0.0
        response.success = True
        response.message = f"计数已归零（之前 {previous} 张，文件未删除）"
        self._publish_status()
        return response


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = DatasetCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node._write_session()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
