"""离线/在线立方体识别命令行工具（不需要 ROS）。

典型用法（**先在没有相机的情况下验证算法**）：

.. code-block:: bash

    # 1) 合成场景自测：不用任何数据，直接看算法能不能检出
    python -m underwater_perception.tools.cube_detect_cli --synthetic --out ./demo_out

    # 2) 单张图/图片文件夹
    python -m underwater_perception.tools.cube_detect_cli --input ./images --out ./demo_out

    # 3) 视频（逐帧检测，统计检出率）
    python -m underwater_perception.tools.cube_detect_cli --input pool.mp4 --sample-every 3

    # 4) 现场调阈值：把调色掩码也存下来看
    python -m underwater_perception.tools.cube_detect_cli --input shot.jpg --print-mask

    # 5) 看生效参数（可复制进 config/cube_detector.yaml）
    python -m underwater_perception.tools.cube_detect_cli --dump-config

输出：``--out`` 目录下的标注图 ``annotated_*.png``、掩码图 ``mask_*.png``、逐帧结果 ``detections.json``。
退出码：0 无异常（即使没检出）；1 输入路径/参数错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..core.cube_geometry import CubeDetector, detect_observations
from ..core.params import CubeDetectorConfig, dump_flat_yaml, load_flat_yaml, resolve_config_path
from ..core.types import DetectionFrame, polygon_iou
from ..core.visualize import annotate_detections, mask_overlay

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _log(message: str) -> None:
    print(message, flush=True)


def _imread(path: Path) -> Optional[np.ndarray]:
    """用 ``imdecode`` 读图，兼容中文/空格路径（OpenCV 在 Windows 上对非 ASCII 路径的经典问题）。"""
    if not path.is_file():
        return None
    buffer = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _imwrite(path: Path, image: np.ndarray) -> None:
    """写图，同样兼容非 ASCII 路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, buffer = cv2.imencode(extension, image)
    if ok:
        buffer.tofile(str(path))


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（默认值来自 ``CubeDetectorConfig``，可被 YAML 覆盖）。"""
    parser = argparse.ArgumentParser(
        description="几何旁路立方体识别（合成数据 / 图片 / 视频）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="检测参数 YAML（config/cube_detector.yaml）")
    parser.add_argument("--input", default="", help="图片文件夹、单张图片或视频文件")
    parser.add_argument("--synthetic", action="store_true", help="用合成水下场景做自测（无需任何数据）")
    parser.add_argument("--synthetic-count", type=int, default=6, help="合成场景数量")
    parser.add_argument("--out", default="", help="输出目录（保存标注图/掩码/JSON）")
    parser.add_argument("--sample-every", type=int, default=1, help="视频模式每隔多少帧处理一帧")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧/图（0=全部）")
    parser.add_argument("--print-mask", action="store_true", help="同时保存调色掩码与叠加图（调阈值用）")
    parser.add_argument("--no-annotate", action="store_true", help="不保存标注图（只看统计）")
    parser.add_argument("--dump-config", action="store_true", help="打印生效参数后退出")
    parser.add_argument("--mode", default="", help="覆盖检测模式: any / square_face / cube_silhouette")
    parser.add_argument("--cube-edge", type=float, default=None, help="覆盖立方体边长（米）")
    parser.add_argument("--no-fit", action="store_true", help="关闭轮廓拟合（更快但斜视测距误差大）")
    parser.add_argument(
        "--track",
        action="store_true",
        help="启用跨帧跟踪（视频/序列用）。默认关闭：独立图片之间毫无关联，"
        "开启后 min_hits 会把每张图的首帧检测全部压掉（实测 6 张图只能报出 3 张）",
    )
    return parser


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> CubeDetectorConfig:
    """YAML 默认值 + 命令行覆盖 -> 检测配置。"""
    flat = CubeDetectorConfig().to_flat()
    resolved = resolve_config_path(args.config or None, "config/cube_detector.yaml")
    if resolved:
        file_values = load_flat_yaml(resolved)
        flat.update({k: v for k, v in file_values.items() if k in flat})
        _log(f"已加载配置: {resolved}")
    if args.mode:
        if args.mode not in ("any", "square_face", "cube_silhouette"):
            parser.error("--mode 只能是 any / square_face / cube_silhouette")
        flat["mode"] = args.mode
    if args.cube_edge is not None:
        flat["cube_edge_m"] = float(args.cube_edge)
    if args.no_fit:
        flat["use_silhouette_fit"] = False
    # 独立图片之间没有时序关系：默认关闭跟踪，否则 min_hits 会压掉首帧检测
    flat["enable_tracking"] = bool(args.track)
    return CubeDetectorConfig.from_flat(flat)


def _synthetic_images(count: int):
    """生成合成水下立方体场景（含真值轮廓，用于计算 IoU）。"""
    from ..core.synthetic import default_camera_matrix, render_cube_scene, render_underwater_background

    image_size = (1280, 480)
    camera = default_camera_matrix(*image_size)
    background = render_underwater_background(*image_size, seed=1)
    poses = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.45)),
        ((0.45, 0.55, 0.20), (0.0, 0.0, 0.45)),
        ((-0.35, 0.40, -0.25), (0.03, -0.02, 0.55)),
        ((0.30, -0.35, 0.15), (-0.04, 0.02, 0.50)),
        ((0.15, 0.20, 0.05), (0.0, 0.0, 0.80)),
        ((0.20, 0.25, 0.10), (0.02, 0.0, 0.35)),
    ]
    scenes = []
    for index in range(max(1, count)):
        rvec, tvec = poses[index % len(poses)]
        scene = render_cube_scene(
            camera, rvec=rvec, tvec=tvec, image_size=image_size, background=background, seed=10 + index
        )
        scenes.append((f"synthetic_{index:02d}", scene.image, scene.silhouette))
    return scenes


def _gather_inputs(args: argparse.Namespace) -> List[tuple]:
    """收集待处理输入：``[(name, image, ground_truth_polygon_or_None), ...]``。"""
    if args.synthetic:
        return _synthetic_images(args.synthetic_count)

    path = Path(args.input).expanduser()
    if not args.input or not path.exists():
        raise FileNotFoundError(f"--input 不存在: {args.input}（可用 --synthetic 做合成自测）")

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if args.max_frames > 0:
            files = files[: args.max_frames]
        items = []
        for file in files:
            image = _imread(file)
            if image is not None:
                items.append((file.name, image, None))
        return items

    if path.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        items = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(1, args.sample_every) == 0:
                items.append((f"{path.stem}_{index:06d}", frame, None))
                if args.max_frames > 0 and len(items) >= args.max_frames:
                    break
            index += 1
        capture.release()
        return items

    image = _imread(path)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return [(path.name, image, None)]


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args, parser)

    if args.dump_config:
        print(dump_flat_yaml(config.to_flat(), "cube_detector"))
        return 0

    try:
        items = _gather_inputs(args)
    except (FileNotFoundError, RuntimeError) as exc:
        _log(f"[错误] {exc}")
        return 1
    if not items:
        _log("[错误] 没有可处理的输入")
        return 1

    detector = CubeDetector(config)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    detected_frames = 0
    ious: List[float] = []
    confidences: List[float] = []
    distances: List[float] = []
    track_ids = set()

    _log(f"处理 {len(items)} 帧 | 模式={config.detect.mode} 颜色={config.color.preset} 最小置信度={config.detect.min_confidence}")
    _log("")

    for name, image, truth in items:
        detections = detector.detect(image)
        if detections:
            detected_frames += 1
            for detection in detections:
                confidences.append(detection.confidence)
                if detection.track_id is not None:
                    track_ids.add(detection.track_id)
                if detection.measurement is not None and detection.measurement.distance_m is not None:
                    distances.append(detection.measurement.distance_m)
            if truth is not None:
                ious.append(polygon_iou(detections[0].polygon(), np.asarray(truth).tolist()))

        frame = DetectionFrame(
            detections=detections,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            source="geometry",
        )
        records.append({"name": name, "frame": frame.to_dict()})

        if out_dir and not args.no_annotate:
            annotated = annotate_detections(image, detections, config=config, mask=None, title=name)
            _imwrite(out_dir / f"annotated_{name}.png", annotated)
        if out_dir and args.print_mask:
            observations, mask = detect_observations(image, config, return_mask=True)
            _imwrite(out_dir / f"mask_{name}.png", mask)
            _imwrite(out_dir / f"mask_overlay_{name}.png", mask_overlay(image, mask))

        detail = ", ".join(
            (f"#{d.track_id} " if d.track_id is not None else "")
            + f"conf={d.confidence:.2f}"
            + (f" {d.measurement.distance_m:.2f}m" if d.measurement and d.measurement.distance_m else "")
            for d in detections
        )
        _log(f"{name:28s} 检出 {len(detections)} 个  {detail}")

    # ---- 统计 ----
    _log("")
    _log("=" * 74)
    _log(f"帧数            : {len(items)}")
    _log(f"有检出的帧      : {detected_frames} ({100.0 * detected_frames / max(len(items), 1):.1f}%)")
    _log(f"轨迹数(track_id): {len(track_ids)}")
    if confidences:
        _log(f"置信度 均值/最小 : {float(np.mean(confidences)):.3f} / {float(np.min(confidences)):.3f}")
    if distances:
        _log(f"距离估计 均值   : {float(np.mean(distances)):.3f} m（若内参与边长正确才有物理意义）")
    if ious:
        _log(f"与合成真值 IoU  : 均值 {float(np.mean(ious)):.3f}，最小 {float(np.min(ious)):.3f}")
    _log("=" * 74)

    if out_dir:
        payload = {
            "config": config.to_flat(),
            "frames": records,
            "summary": {
                "frames": len(items),
                "detected_frames": detected_frames,
                "mean_confidence": float(np.mean(confidences)) if confidences else None,
                "mean_iou_vs_synthetic_truth": float(np.mean(ious)) if ious else None,
                "track_ids": sorted(track_ids),
            },
        }
        (out_dir / "detections.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"结果已写入: {out_dir / 'detections.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
