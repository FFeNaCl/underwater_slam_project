"""离线标定命令行工具（不需要 ROS，不需要相机）。

四种数据来源，参数全部可通过命令行或 YAML 覆盖：

.. code-block:: bash

    # 1) 图片文件夹（最常用，也最稳）
    python -m underwater_perception.tools.calibrate_cli --images ./calib_images --config config/calibration.yaml

    # 2) 视频文件
    python -m underwater_perception.tools.calibrate_cli --video pool.mp4 --sample-every 15

    # 3) 直接读 USB 相机（Linux 上 /dev/video0 -> --camera 0）
    python -m underwater_perception.tools.calibrate_cli --camera 0 --capture 20 --interval 1.5

    # 4) 从 ROS 话题在线采集（需要装了 ROS2 的机器）
    python -m underwater_perception.tools.calibrate_cli --ros-topic /image_enhanced --capture 20

    # 只看参数、不标定
    python -m underwater_perception.tools.calibrate_cli --dump-config

输出（写到 ``--out`` 目录）::

    ost.yaml                     ROS camera_calibration 兼容
    <orbslam3_name>.yaml         ORB-SLAM3 单目设置（放到 orb_slam3/config/Monocular/）
    enhancer_camera_params.yaml  图像增强节点的 camera_matrix / distortion_coeffs
    calibration_report.json      全部指标与警告
    undistorted/                 去畸变示例图（可选）

退出码：0 成功；2 有效视图不足；3 其它错误。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from ..core.calibration import (
    CalibrationResult,
    ViewResult,
    calibrate,
    detect_pattern,
    undistort_image,
    view_from_array,
    view_from_path,
    write_all_outputs,
)
from ..core.camera_io import DEFAULT_FOURCC, DEFAULT_FPS, camera_info, open_camera
from ..core.params import CalibrationConfig, dump_flat_yaml, load_flat_yaml, resolve_config_path
from ..core.synthetic import chessboard_object_points

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")


def _log(message: str) -> None:
    print(message, flush=True)


def _add_calibration_arguments(parser: argparse.ArgumentParser, defaults: Optional[dict] = None) -> None:
    """把标定参数挂到命令行。

    ``defaults`` 来自 ``--config`` 指向的 YAML（没有就用 :class:`CalibrationConfig` 的代码默认值）。
    这样优先级是：**命令行 > YAML > 代码默认值**，符合直觉且不需要"猜用户是否显式传了参数"。
    """
    base = CalibrationConfig()
    values = dict(defaults or {})

    def pick(key: str, fallback):
        return values.get(key, fallback)

    parser.add_argument("--config", default="", help="标定参数 YAML（可选，键名与 config/calibration.yaml 一致）")
    parser.add_argument(
        "--pattern", default=pick("pattern", base.pattern), choices=["chessboard", "circles", "charuco"]
    )
    parser.add_argument("--cols", type=int, default=pick("cols", base.cols), help="内角点列数（8x6 棋盘格填 8）")
    parser.add_argument("--rows", type=int, default=pick("rows", base.rows), help="内角点行数（8x6 棋盘格填 6）")
    parser.add_argument("--square", type=float, default=pick("square_size", base.square_size), help="方格边长（米）")
    parser.add_argument("--min-views", type=int, default=pick("min_views", base.min_views))
    parser.add_argument("--max-views", type=int, default=pick("max_views", base.max_views))
    parser.add_argument("--min-corners-ratio", type=float, default=pick("min_corners_ratio", base.min_corners_ratio))
    parser.add_argument("--coverage-grid", type=int, default=pick("coverage_grid", base.coverage_grid))
    parser.add_argument("--min-coverage-cells", type=int, default=pick("min_coverage_cells", base.min_coverage_cells))
    parser.add_argument("--subpix-window", type=int, default=pick("subpix_window", base.subpix_window))
    parser.add_argument("--fix-k3", action="store_true", default=pick("fix_k3", base.fix_k3), help="固定 k3=0（畸变很小时更稳）")
    parser.add_argument("--fix-aspect-ratio", action="store_true", default=pick("fix_aspect_ratio", base.fix_aspect_ratio))
    parser.add_argument("--zero-tangent-dist", action="store_true", default=pick("zero_tangent_dist", base.zero_tangent_dist))
    parser.add_argument("--undistort-alpha", type=float, default=pick("undistort_alpha", base.undistort_alpha))
    parser.add_argument("--camera-name", default=pick("camera_name", base.camera_name))
    parser.add_argument("--orbslam3-name", default=pick("orbslam3_settings_name", base.orbslam3_settings_name))
    parser.add_argument("--fps", type=float, default=pick("fps", base.fps), help="写进 ORB-SLAM3 配置的 Camera.fps")
    parser.add_argument("--out", default=pick("output_dir", base.output_dir), help="输出目录")
    parser.add_argument("--save-undistorted", action="store_true", default=pick("save_undistorted", base.save_undistorted))
    parser.add_argument("--no-save-undistorted", dest="save_undistorted", action="store_false")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """数据来源相关参数。"""
    parser.add_argument("--images", default="", help="图片文件夹（按文件名排序）")
    parser.add_argument("--video", default="", help="视频文件")
    parser.add_argument("--camera", type=int, default=-1, help="USB/CSI 相机序号（Linux 上 /dev/video0 -> 0；video1/video3 常是 metadata，别用）")
    parser.add_argument("--fourcc", default=DEFAULT_FOURCC, choices=["MJPG", "YUYV"], help="像素格式，MJPG 才有 30 fps")
    parser.add_argument("--camera-fps", type=float, default=DEFAULT_FPS, help="相机目标帧率")
    parser.add_argument("--ros-topic", default="", help="ROS2 图像话题（需要 rclpy）")
    parser.add_argument("--capture", type=int, default=20, help="实时采集的目标视图数（camera / ros-topic 模式）")
    parser.add_argument("--interval", type=float, default=1.0, help="实时采集的尝试间隔（秒）")
    parser.add_argument("--sample-every", type=int, default=10, help="视频模式每隔多少帧取一帧")
    parser.add_argument("--max-images", type=int, default=0, help="图片模式最多读取多少张（0=全部）")
    parser.add_argument("--dump-config", action="store_true", help="打印生效参数后退出")


def _config_from_args(args: argparse.Namespace) -> CalibrationConfig:
    """命令行命名空间 -> :class:`CalibrationConfig`（默认值已在解析阶段合并过 YAML）。"""
    return CalibrationConfig.from_flat(
        {
            "pattern": args.pattern,
            "cols": args.cols,
            "rows": args.rows,
            "square_size": args.square,
            "min_views": args.min_views,
            "max_views": args.max_views,
            "min_corners_ratio": args.min_corners_ratio,
            "coverage_grid": args.coverage_grid,
            "min_coverage_cells": args.min_coverage_cells,
            "subpix_window": args.subpix_window,
            "fix_k3": args.fix_k3,
            "fix_aspect_ratio": args.fix_aspect_ratio,
            "zero_tangent_dist": args.zero_tangent_dist,
            "undistort_alpha": args.undistort_alpha,
            "camera_name": args.camera_name,
            "output_dir": args.out,
            "orbslam3_settings_name": args.orbslam3_name,
            "fps": args.fps,
            "save_undistorted": args.save_undistorted,
            "save_debug_corners": True,
        }
    )


def _collect_from_images(args: argparse.Namespace, config: CalibrationConfig):
    """从图片文件夹采集视图。"""
    directory = Path(args.images).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"--images 目录不存在: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_images > 0:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"目录里没有图片: {directory}")
    _log(f"从 {directory} 读取 {len(paths)} 张图片")
    return [view_from_path(path, config) for path in paths]


def _collect_from_video(args: argparse.Namespace, config: CalibrationConfig):
    """从视频文件采集视图。"""
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")
    views: List[ViewResult] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % max(1, args.sample_every) == 0:
            views.append(view_from_array(frame, config, source=f"{Path(args.video).name}#{index}"))
        index += 1
    capture.release()
    _log(f"视频共 {index} 帧，按每 {args.sample_every} 帧取一帧，得到 {len(views)} 个候选视图")
    return views


def _collect_from_camera(args: argparse.Namespace, config: CalibrationConfig):
    """从 USB/CSI 相机交互式采集（按 q 结束，按空格手动采一帧）。"""
    capture = open_camera(
        int(args.camera),
        width=0,
        height=0,
        fourcc=str(args.fourcc),
        fps=float(args.camera_fps),
    )
    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开相机 {args.camera}（检查 /dev/video* 权限与占用；"
            "Linux 下 /dev/video1、/dev/video3 常是 metadata 节点，请改用 0/2）"
        )
    info = camera_info(capture)
    _log("相机实际格式: {fourcc} {width}x{height} @ {fps:.1f} fps".format(**info))
    _log("相机已打开：把标定板放到画面中不同位置/角度，按【空格】采集，按【q】结束")
    views: List[ViewResult] = []
    try:
        while len(views) < args.capture:
            ok, frame = capture.read()
            if not ok:
                break
            view = view_from_array(frame, config, source=f"camera{args.camera}#{len(views)}")
            preview = frame.copy()
            if view.found:
                cv2.drawChessboardCorners(preview, config.pattern_size, view.corners, view.accepted)
            label = f"views={len(views)}/{args.capture} found={view.found}"
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 230, 255), 2, cv2.LINE_AA)
            cv2.imshow("calibrate_cli", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if view.accepted:
                    views.append(view)
                    _log(f"采集 {len(views)}/{args.capture}")
                else:
                    _log("当前帧未检出棋盘或不满足角点比例要求，未采集")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return views


def _collect_from_ros(args: argparse.Namespace, config: CalibrationConfig, timeout_s: float = 120.0):
    """从 ROS2 话题在线采集（延迟 import rclpy，保证无 ROS 机器也能用其它模式）。"""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image as RosImage
    except ImportError as exc:  # pragma: no cover - 无 ROS 环境
        raise RuntimeError(f"--ros-topic 需要 ROS2 环境（rclpy 导入失败: {exc}）") from exc

    from ..nodes.image_bridge import msg_to_bgr

    views: List[ViewResult] = []
    rclpy.init()
    node = Node("calibrate_cli")
    state = {"last": 0.0}

    def on_image(msg) -> None:
        import time

        now = time.time()
        if now - state["last"] < args.interval:
            return
        try:
            image = msg_to_bgr(msg)
        except Exception:
            return
        view = view_from_array(image, config, source=f"ros#{len(views)}")
        if view.accepted:
            state["last"] = now
            views.append(view)
            _log(f"采集 {len(views)}/{args.capture}（角点占比 {view.corner_ratio:.2f}）")

    node.create_subscription(RosImage, args.ros_topic, on_image, 5)
    _log(f"订阅 {args.ros_topic} 采集中（最多 {args.capture} 个视图，超时 {timeout_s:.0f} s）")
    try:
        import time

        deadline = time.time() + timeout_s
        while len(views) < args.capture and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return views


def _report(result: CalibrationResult, config: CalibrationConfig, views: Iterable[ViewResult]) -> None:
    """打印人类可读的标定报告。"""
    _log("")
    _log("=" * 74)
    _log("标定结果")
    _log("=" * 74)
    if not result.ok:
        _log(f"失败: {result.message}")
        return
    _log(f"有效视图      : {result.n_views_used}/{result.n_views_total}")
    _log(f"RMS 重投影误差: {result.rms:.4f} px（越小越好，<0.5 优秀，>1.0 需检查）")
    _log(f"内参 fx,fy    : {result.fx:.3f}, {result.fy:.3f} px")
    _log(f"主点 cx,cy    : {result.cx:.3f}, {result.cy:.3f} px（图像中心 {(result.image_size[0]-1)/2:.1f}, {(result.image_size[1]-1)/2:.1f}）")
    _log(f"畸变 k1,k2,p1,p2,k3: {', '.join(f'{v:.5f}' for v in result.distortion_list(5))}")
    _log(f"质量等级      : {result.quality_grade()}")
    worst = sorted(zip(result.per_view_errors, [v.source for v in views if v.found]), reverse=True)[:3]
    if worst:
        _log("误差最大的视图:")
        for error, source in worst:
            _log(f"  {error:7.3f} px  {source}")
    for warning in result.warnings:
        _log(f"[警告] {warning}")
    _log("")


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    # 第一段：只解析 --config，用 YAML 的值当作完整解析器的默认值（命令行仍可覆盖）
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args(argv)
    file_defaults: dict = {}
    resolved = resolve_config_path(known.config or None, None)
    if resolved:
        file_defaults = load_flat_yaml(resolved)
        _log(f"已加载配置: {resolved}")

    parser = argparse.ArgumentParser(
        description="单目相机标定（棋盘格 8x6 / 25mm 为默认值，参数可改）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_calibration_arguments(parser, file_defaults)
    _add_source_arguments(parser)
    args = parser.parse_args(argv)

    config = _config_from_args(args)
    if args.dump_config:
        _log(dump_flat_yaml(
            {
                "pattern": config.pattern,
                "cols": config.cols,
                "rows": config.rows,
                "square_size": config.square_size,
                "min_views": config.min_views,
                "max_views": config.max_views,
                "min_corners_ratio": config.min_corners_ratio,
                "coverage_grid": config.coverage_grid,
                "subpix_window": config.subpix_window,
                "fix_k3": config.fix_k3,
                "undistort_alpha": config.undistort_alpha,
                "camera_name": config.camera_name,
                "output_dir": config.output_dir,
                "orbslam3_settings_name": config.orbslam3_settings_name,
                "fps": config.fps,
            },
            "calibration_node",
        ))
        return 0

    # ---- 采集 ----
    try:
        if args.images:
            views = _collect_from_images(args, config)
        elif args.video:
            views = _collect_from_video(args, config)
        elif args.camera >= 0:
            views = _collect_from_camera(args, config)
        elif args.ros_topic:
            views = _collect_from_ros(args, config)
        else:
            parser.error("必须指定数据来源之一：--images / --video / --camera / --ros-topic")
            return 3
    except (FileNotFoundError, RuntimeError) as exc:
        _log(f"[错误] {exc}")
        return 3

    accepted = [v for v in views if v.accepted]
    skipped = [v for v in views if not v.accepted]
    _log(f"检出角点的视图: {len(accepted)}，未通过的: {len(skipped)}")
    if skipped:
        reasons: dict = {}
        for view in skipped:
            reasons[view.reason or view.method] = reasons.get(view.reason or view.method, 0) + 1
        _log(f"未通过原因统计: {reasons}")

    if len(accepted) < 3:
        _log("[错误] 有效视图不足 3 个，无法标定。请检查：图案尺寸参数、方格边长、棋盘是否完整入画、对比度是否足够")
        return 2

    image_size = accepted[0].image_size
    mismatched = [v for v in accepted if v.image_size != image_size]
    if mismatched:
        _log(f"[警告] 有 {len(mismatched)} 个视图尺寸与首个视图不同，已忽略它们（标定要求同尺寸）")
        accepted = [v for v in accepted if v.image_size == image_size]

    # ---- 标定 ----
    result = calibrate(accepted, image_size, config, source="cli")
    _report(result, config, views)
    if not result.ok:
        return 3 if result.message != "not_enough_views" else 2

    outputs = write_all_outputs(args.out, result, config, views=views)
    _log("输出文件:")
    for name, path in outputs.items():
        _log(f"  {name:9s} -> {path}")

    if config.save_undistorted:
        directory = Path(args.out) / "undistorted"
        directory.mkdir(parents=True, exist_ok=True)
        saved = 0
        for index, view in enumerate(accepted[:5]):
            if isinstance(view.source, str) and Path(view.source).is_file():
                buffer = np.fromfile(view.source, dtype=np.uint8)
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                undistorted, _, _ = undistort_image(image, result, config.undistort_alpha)
                target = directory / f"undistorted_{index:02d}.jpg"
                cv2.imwrite(str(target), undistorted)
                saved += 1
        if saved:
            _log(f"去畸变示例图: {directory}（{saved} 张，用于人眼检查边缘是否变直）")

    _log("")
    _log("下一步（照着做即可）：")
    _log(f"  1) 把 {Path(args.out) / (config.orbslam3_settings_name + '.yaml')} 复制到")
    _log("     src/ros2_orb_slam3/orb_slam3/config/Monocular/ 下")
    _log(f"  2) 把 {Path(args.out) / 'enhancer_camera_params.yaml'} 里的 camera_matrix / distortion_coeffs")
    _log("     填进 config/enhancer_underwater.yaml 并设 use_undistort: true")
    _log("  3) 重启 SLAM 链路，用 ros2 topic echo /slam/status --once 确认跟踪状态")
    return 0


if __name__ == "__main__":
    sys.exit(main())
