"""标定节点：在 ROS 里边采图边标定（也可离线用 ``calibrate_cli``）。

与 ``camera_calibration`` 的区别：本节点把标定参数全部显式暴露为 ROS 参数，
并把结果直接写成项目需要的四件套（ost.yaml / ORB-SLAM3 设置 / 增强节点参数 / JSON 报告），
因此不需要再手工搬运数字。

服务
----
``~/capture``  (``std_srvs/Trigger``)  人工采一帧（当前帧检测到棋盘才会计入）
``~/run``      (``std_srvs/Trigger``)  执行标定并写出四件套
``~/reset``    (``std_srvs/Trigger``)  清空已采集视图

话题
----
订阅 ``image_topic``（默认 ``/image_enhanced``）
发布 ``debug_topic``（默认 ``/calibration/debug``，标出检出角点）
发布 ``status_topic``（默认 ``/calibration/status``，JSON 状态）
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ..core.calibration import CalibrationResult, ViewResult, calibrate, detect_pattern, write_all_outputs
from ..core.params import CalibrationConfig, declare_from_flat, load_flat_yaml, resolve_config_path
from ..core.synthetic import chessboard_object_points
from .image_bridge import bgr_to_msg, msg_to_bgr, stamp_to_seconds

DEFAULT_CONFIG = "config/calibration.yaml"


class CalibrationNode(Node):
    """采集棋盘视图 -> 标定 -> 输出四件套。"""

    def __init__(self) -> None:
        super().__init__("calibration_node")

        self.declare_parameter("config_file", "")
        config_file = str(self.get_parameter("config_file").value)
        resolved = resolve_config_path(config_file or None, DEFAULT_CONFIG)

        flat_defaults = {
            "pattern": CalibrationConfig().pattern,
            "cols": CalibrationConfig().cols,
            "rows": CalibrationConfig().rows,
            "square_size": CalibrationConfig().square_size,
            "min_views": CalibrationConfig().min_views,
            "max_views": CalibrationConfig().max_views,
            "min_corners_ratio": CalibrationConfig().min_corners_ratio,
            "coverage_grid": CalibrationConfig().coverage_grid,
            "min_coverage_cells": CalibrationConfig().min_coverage_cells,
            "subpix_window": CalibrationConfig().subpix_window,
            "fix_k3": CalibrationConfig().fix_k3,
            "fix_aspect_ratio": CalibrationConfig().fix_aspect_ratio,
            "zero_tangent_dist": CalibrationConfig().zero_tangent_dist,
            "undistort_alpha": CalibrationConfig().undistort_alpha,
            "camera_name": CalibrationConfig().camera_name,
            "output_dir": CalibrationConfig().output_dir,
            "orbslam3_settings_name": CalibrationConfig().orbslam3_settings_name,
            "fps": CalibrationConfig().fps,
            "save_undistorted": CalibrationConfig().save_undistorted,
            "save_debug_corners": CalibrationConfig().save_debug_corners,
        }
        if resolved:
            flat_defaults.update({k: v for k, v in load_flat_yaml(resolved).items() if k in flat_defaults})
            self.get_logger().info(f"已加载配置文件: {resolved}")

        flat_defaults.update(
            {
                "image_topic": "/image_enhanced",
                "debug_topic": "/calibration/debug",
                "status_topic": "/calibration/status",
                "auto_capture": True,
                "capture_interval_s": 1.0,
                "publish_debug": True,
            }
        )
        declare_from_flat(self, flat_defaults, skip=("config_file",))

        flat = {key: self.get_parameter(key).value for key in flat_defaults}
        self.config = CalibrationConfig.from_flat(flat)
        self.image_topic = str(flat["image_topic"])
        self.debug_topic = str(flat["debug_topic"])
        self.status_topic = str(flat["status_topic"])
        self.auto_capture = bool(flat["auto_capture"])
        self.capture_interval = float(flat["capture_interval_s"])
        self.publish_debug = bool(flat["publish_debug"])

        self.views: List[ViewResult] = []
        self.rejected = 0
        self.image_size = (0, 0)
        self.last_result: Optional[CalibrationResult] = None
        self.last_capture_time = 0.0
        #: 最近一帧图像与它的检出角点，供 ``~/capture`` 手动采集使用
        self._last_image: Optional[np.ndarray] = None
        self._last_corners: Optional[np.ndarray] = None
        self._last_method = "NONE"

        self.subscription = self.create_subscription(Image, self.image_topic, self.on_image, 5)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 5) if self.publish_debug else None
        self.status_pub = self.create_publisher(String, self.status_topic, 5)

        self.create_service(Trigger, "~/capture", self.on_capture)
        self.create_service(Trigger, "~/run", self.on_run)
        self.create_service(Trigger, "~/reset", self.on_reset)

        self.get_logger().info(
            "标定节点启动 | 输入 %s | 图案 %s %dx%d 方格 %.1f mm | 目标视图 >= %d | 输出目录 %s"
            % (
                self.image_topic,
                self.config.pattern,
                self.config.cols,
                self.config.rows,
                self.config.square_size * 1000.0,
                self.config.min_views,
                self.config.output_dir,
            )
        )
        self.get_logger().info(
            "采集方式: "
            + ("自动（每 %.1f s 尝试一帧）" % self.capture_interval if self.auto_capture else "手动（调用 ~/capture 服务）")
        )

    # ------------------------------------------------------------------ 回调
    def on_image(self, msg: Image) -> None:
        """检测棋盘：自动模式下按间隔采集，并发布调试图与状态。"""
        try:
            image = msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"图像转换失败: {exc}", throttle_duration_sec=5.0)
            return

        self.image_size = (int(image.shape[1]), int(image.shape[0]))
        corners, method = detect_pattern(image, self.config)
        self._last_image = image
        self._last_corners = corners
        self._last_method = method

        now = time.time()
        if corners is None:
            self.rejected += 1
        elif self.auto_capture and (now - self.last_capture_time) >= self.capture_interval:
            if self._add_view_from_detection(corners, method, str(int(msg.header.stamp.sec))):
                self.last_capture_time = now

        if self.debug_pub is not None:
            preview = image.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, self.config.pattern_size, corners, True)
            text = f"views={len(self.views)}/{self.config.max_views} rejected={self.rejected}"
            cv2.putText(preview, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 230, 255), 2, cv2.LINE_AA)
            self.debug_pub.publish(bgr_to_msg(preview, msg.header.stamp, msg.header.frame_id))

        status = String()
        status.data = json.dumps(
            {
                "views": len(self.views),
                "rejected": self.rejected,
                "max_views": int(self.config.max_views),
                "pattern_found": corners is not None,
                "method": method,
                "ready": len(self.views) >= int(self.config.min_views),
                "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            },
            ensure_ascii=True,
        )
        self.status_pub.publish(status)

    # ------------------------------------------------------------------ 服务
    def on_capture(self, request, response):
        """人工采集当前帧（要求该帧能检出棋盘）。"""
        if self._last_image is None:
            response.success = False
            response.message = "尚未收到任何图像"
            return response
        if self._last_corners is None:
            response.success = False
            response.message = "当前帧未检出棋盘格，请把标定板完整放入画面后再试"
            return response
        if len(self.views) >= int(self.config.max_views):
            response.success = False
            response.message = f"已达视图上限 {self.config.max_views}，如需继续请先调用 ~/reset"
            return response
        added = self._add_view_from_detection(self._last_corners, self._last_method, "manual")
        self.last_capture_time = time.time()
        response.success = bool(added)
        response.message = (
            f"已采集第 {len(self.views)} 个视图" if added else "采集失败"
        )
        return response

    def on_run(self, request, response):
        """执行标定并写出四件套输出文件。"""
        if len(self.views) < 3:
            response.success = False
            response.message = f"视图不足（{len(self.views)} 个），至少需要 3 个，推荐 {self.config.min_views} 个以上"
            return response
        if min(self.image_size) <= 0:
            response.success = False
            response.message = "图像尺寸未知"
            return response

        result = calibrate(self.views, self.image_size, self.config, source="ros")
        self.last_result = result
        if not result.ok:
            response.success = False
            response.message = f"标定失败: {result.message}"
            self.get_logger().error(response.message)
            return response

        outputs = write_all_outputs(self.config.output_dir, result, self.config, views=self.views)
        response.success = True
        response.message = (
            f"标定完成: 视图 {result.n_views_used}/{result.n_views_total}, RMS={result.rms:.4f} px, "
            f"等级={result.quality_grade()}, 输出: {outputs}"
        )
        self.get_logger().info(response.message)
        for warning in result.warnings:
            self.get_logger().warning(warning)
        return response

    def on_reset(self, request, response):
        """清空已采集视图（换水质/换位置后重新采集）。"""
        count = len(self.views)
        self.views.clear()
        self.rejected = 0
        response.success = True
        response.message = f"已清空 {count} 个视图"
        return response

    # ------------------------------------------------------------------ 内部
    def _add_view_from_detection(self, corners, method: str, source: str) -> bool:
        """由一次成功检出构造 :class:`ViewResult` 并入库。

        **关键点**：``object_points`` 必须在这里补上——``calibrate()`` 会过滤掉
        ``object_points is None`` 的视图，如果漏了这一步，节点"采集成功但标定时
        一个视图都用不上"（开发过程中真实踩过）。
        """
        limit = int(self.config.max_views)
        if len(self.views) >= limit:
            return False
        expected = int(self.config.cols) * int(self.config.rows)
        view = ViewResult(
            corners=corners,
            object_points=chessboard_object_points(self.config.cols, self.config.rows, self.config.square_size),
            image_size=self.image_size,
            source=source,
            method=method,
            corner_ratio=float(len(corners)) / float(expected) if expected else 0.0,
            coverage_cells=0,
            accepted=True,
        )
        self.views.append(view)
        self.get_logger().info(
            f"采集视图 {len(self.views)}/{limit}: method={method} 角点占比={view.corner_ratio:.2f}"
        )
        if len(self.views) == int(self.config.min_views):
            self.get_logger().info(
                f"已达到建议视图数 {self.config.min_views}，可调用 ~/run 执行标定（继续采集可提高精度）"
            )
        return True


def main(args=None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
