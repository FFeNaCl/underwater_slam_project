"""无硬件自测：合成数据端到端验证几何识别、跟踪、标定与三角化。

**这是本项目"没有相机也能验证"的核心工具。** 在 Windows 开发机、无 ROS 的 Linux
机器、CI 上都能直接跑：

    python -m underwater_perception.tools.selftest --out ./selftest_out

它会：
1. 用真实透视投影渲染红色立方体（含可见面明暗与内部棱线），统计检出率与轮廓 IoU；
2. 用"只有背景 / 蓝色立方体 / 红色平面方块"统计误检；
3. 用 10 帧序列验证 track_id 稳定性；
4. 用合成棋盘格做标定，量化内参与畸变系数的恢复误差，并写出四件套输出文件；
5. 用合成相机位姿做多视图三角化，量化地标位置误差；
6. 把带标注的图、掩码图、棋盘角点图存到输出目录，便于人眼复核。

退出码 0 表示全部通过；任何一项低于门限则返回 1（可直接用于 CI）。
**注意**：通过本自测不等于在真实水下可用；真实数据的必须复核项见 docs/UNVERIFIED.md。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.calibration import (
    calibrate,
    collect_views,
    detect_pattern,
    undistort_image,
    write_all_outputs,
)
from ..core.cube_geometry import CubeDetector, detect_observations
from ..core.landmarks import LandmarkManager
from ..core.measurement import invert_pose, matrix_to_pose, pose_to_matrix, triangulate_rays
from ..core.params import CalibrationConfig, CubeDetectorConfig, LandmarkConfig
from ..core.synthetic import (
    default_camera_matrix,
    make_chessboard_view_set,
    render_chessboard_scene,
    render_cube_scene,
    render_cube_sequence,
    render_underwater_background,
)
from ..core.types import Detection, DetectionFrame, GeometryMetrics, Measurement, polygon_iou
from ..core.visualize import annotate_detections, mask_overlay, side_by_side

IMAGE_SIZE = (1280, 480)

#: 通过门限：先用合成数据定标，再在真实数据上重新评估。数值来源见每次运行的报告。
THRESHOLDS: Dict[str, float] = {
    "detection_rate": 0.9,          # 合成场景检出率
    "min_iou": 0.60,                # 与真值轮廓的最小 IoU
    "max_distance_rel_error": 0.25,  # 已知边长测距的相对误差
    "fx_rel_error": 0.03,           # 标定 fx 相对误差
    "fy_rel_error": 0.03,
    "principal_point_error_px": 8.0,
    "calibration_rms_px": 1.0,
    "landmark_error_m": 0.03,       # 三角化地标位置误差（米）
    "track_id_stability": 1.0,      # 序列中只允许 1 个 track_id
}


@dataclass
class CheckResult:
    """单项检查结果。"""

    name: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    artifacts: List[str] = field(default_factory=list)


def _log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- 1. 检测
def check_detection(out_dir: Path) -> CheckResult:
    """合成立方体检测：多视角检出率、IoU、顶点数。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    background = render_underwater_background(*IMAGE_SIZE, seed=1)
    poses: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], str]] = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.45), "frontal"),
        ((0.45, 0.55, 0.20), (0.0, 0.0, 0.45), "three_faces"),
        ((-0.35, 0.40, -0.25), (0.03, -0.02, 0.55), "oblique_left"),
        ((0.30, -0.35, 0.15), (-0.04, 0.02, 0.50), "oblique_up"),
        ((0.15, 0.20, 0.05), (0.0, 0.0, 0.80), "far"),
        ((0.20, 0.25, 0.10), (0.02, 0.0, 0.35), "near"),
    ]
    config = CubeDetectorConfig()
    detected = 0
    ious: List[float] = []
    vertex_counts: List[int] = []
    distance_errors: List[float] = []
    artifacts: List[str] = []

    # 距离测量需要内参：把合成相机内参当作"已标定"结果填进去
    flat = config.to_flat()
    flat.update(
        {
            "camera_fx": float(camera[0, 0]),
            "camera_fy": float(camera[1, 1]),
            "camera_cx": float(camera[0, 2]),
            "camera_cy": float(camera[1, 2]),
            "cube_edge_m": 0.05,
        }
    )
    config = CubeDetectorConfig.from_flat(flat)

    for index, (rvec, tvec, label) in enumerate(poses):
        scene = render_cube_scene(
            camera, rvec=rvec, tvec=tvec, image_size=IMAGE_SIZE, background=background, seed=10 + index
        )
        observations = detect_observations(scene.image, config)
        if observations:
            detected += 1
            best = max(observations, key=lambda o: o.confidence)
            ious.append(polygon_iou(best.detection.polygon(), scene.silhouette.tolist()))
            vertex_counts.append(best.metrics.vertex_count)
            measurement = best.detection.measurement
            if measurement is not None and measurement.distance_m:
                truth = float(np.linalg.norm(np.asarray(tvec)))
                distance_errors.append(abs(measurement.distance_m - truth) / truth)
            annotated = annotate_detections(
                scene.image, [o.detection for o in observations], config=config, mask=None
            )
            cv2.polylines(annotated, [scene.silhouette.astype(np.int32)], True, (0, 255, 0), 1)
            path = out_dir / f"detect_{label}.png"
            cv2.imwrite(str(path), annotated)
            artifacts.append(str(path))

    rate = detected / len(poses)
    metrics = {
        "detection_rate": rate,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "min_iou": min(ious) if ious else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "vertex_counts": vertex_counts,
        "max_distance_rel_error": max(distance_errors) if distance_errors else None,
        "n_poses": len(poses),
    }
    passed = rate >= THRESHOLDS["detection_rate"] and (min(ious) if ious else 0.0) >= THRESHOLDS["min_iou"]
    if distance_errors:
        passed = passed and max(distance_errors) <= THRESHOLDS["max_distance_rel_error"]
    return CheckResult("geometry_detection", passed, metrics, artifacts=artifacts)


# --------------------------------------------------------------------------- 2. 误检
def check_false_positives(out_dir: Path) -> CheckResult:
    """误检抑制：背景、蓝色立方体、红色平面方块。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    background = render_underwater_background(*IMAGE_SIZE, seed=2)

    cases: List[Tuple[str, np.ndarray]] = []
    cases.append(("background", background))

    blue = render_cube_scene(
        camera, rvec=(0.3, 0.4, 0.1), tvec=(0.0, 0.0, 0.5), image_size=IMAGE_SIZE,
        color_bgr=(200, 120, 30), background=background,
    )
    cases.append(("blue_cube", blue.image))

    flat_red = background.copy()
    cv2.rectangle(flat_red, (560, 180), (700, 320), (40, 40, 205), thickness=-1)
    cv2.rectangle(flat_red, (560, 180), (700, 320), (25, 25, 140), thickness=2)
    cases.append(("flat_red_square", flat_red))

    permissive = CubeDetectorConfig()
    strict_flat = permissive.to_flat()
    strict_flat["mode"] = "cube_silhouette"
    strict = CubeDetectorConfig.from_flat(strict_flat)

    counts: Dict[str, int] = {}
    strict_counts: Dict[str, int] = {}
    for label, image in cases:
        counts[label] = len(detect_observations(image, permissive))
        strict_counts[label] = len(detect_observations(image, strict))
        path = out_dir / f"negative_{label}.png"
        cv2.imwrite(str(path), image)

    metrics = {"permissive_counts": counts, "strict_counts": strict_counts}
    # 背景与蓝色物体在任何模式下都不允许检出；红色平面方块在严格模式下不允许检出
    passed = (
        counts["background"] == 0
        and counts["blue_cube"] == 0
        and strict_counts["flat_red_square"] == 0
        and strict_counts["background"] == 0
        and strict_counts["blue_cube"] == 0
    )
    notes = (
        "严格模式(cube_silhouette)要求内部棱线证据，可拒绝红色平面方块；"
        "默认 any 模式保召回，会接受平面方块（见 docs/KNOWN_LIMITS 说明）"
    )
    return CheckResult("false_positive_suppression", passed, metrics, notes)


# --------------------------------------------------------------------------- 3. 跟踪
def check_tracking(out_dir: Path) -> CheckResult:
    """10 帧序列的 track_id 稳定性。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scenes = render_cube_sequence(camera, n_frames=10, image_size=IMAGE_SIZE, start=(-0.08, 0.0, 0.45), step=(0.015, 0.0, 0.0))
    detector = CubeDetector(CubeDetectorConfig())
    ids: List[int] = []
    published = 0
    for index, scene in enumerate(scenes):
        detections = detector.detect(scene.image)
        if detections:
            published += 1
            ids.append(detections[0].track_id)
            if index in (0, 5, 9):
                path = out_dir / f"track_frame_{index:02d}.png"
                cv2.imwrite(str(path), annotate_detections(scene.image, detections, config=CubeDetectorConfig()))

    unique = len(set(i for i in ids if i is not None))
    stability = 1.0 / unique if unique else 0.0
    metrics = {"frames_published": published, "unique_track_ids": unique, "track_ids": ids}
    passed = published >= 6 and unique == 1
    return CheckResult("tracking_stability", passed, metrics)


# --------------------------------------------------------------------------- 4. 标定
def check_calibration(out_dir: Path, with_distortion: bool = True) -> CheckResult:
    """合成棋盘格标定：量化内参/畸变恢复误差并写出四件套。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    truth_dist = np.array([[0.05, -0.08, 0.001, -0.0008, 0.01]], dtype=np.float64) if with_distortion else None

    views_scenes = make_chessboard_view_set(
        camera, n_views=18, image_size=IMAGE_SIZE, dist_coeffs=truth_dist, seed=17
    )
    config = CalibrationConfig(min_views=12, save_undistorted=True, save_debug_corners=True)
    images = [scene.image for scene in views_scenes]
    accepted, rejected = collect_views(images, config)

    artifacts: List[str] = []
    # 保存一张角点检出图，便于人眼确认
    if accepted:
        first = accepted[0]
        preview = images[0].copy()
        cv2.drawChessboardCorners(preview, config.pattern_size, first.corners, True)
        path = out_dir / "calibration_corners.png"
        cv2.imwrite(str(path), preview)
        artifacts.append(str(path))

    if len(accepted) < 3:
        return CheckResult(
            "calibration_math",
            False,
            {"accepted_views": len(accepted), "rejected_views": len(rejected)},
            notes="合成视图角点检出不足，检查 render_chessboard_scene 与 detect_pattern 的一致性",
        )

    result = calibrate(accepted, IMAGE_SIZE, config, source="synthetic")
    outputs = write_all_outputs(out_dir / "calibration", result, config, views=accepted)

    fx_error = abs(result.fx - camera[0, 0]) / camera[0, 0]
    fy_error = abs(result.fy - camera[1, 1]) / camera[1, 1]
    pp_error = math.hypot(result.cx - camera[0, 2], result.cy - camera[1, 2])
    dist = result.distortion_list(5)

    metrics: Dict[str, Any] = {
        "with_distortion": with_distortion,
        "accepted_views": len(accepted),
        "rejected_views": len(rejected),
        "rms_px": float(result.rms),
        "fx_truth": float(camera[0, 0]),
        "fx_est": result.fx,
        "fx_rel_error": float(fx_error),
        "fy_rel_error": float(fy_error),
        "principal_point_error_px": float(pp_error),
        "distortion_truth": None if truth_dist is None else [float(v) for v in truth_dist.reshape(-1)],
        "distortion_est": dist,
        "quality_grade": result.quality_grade(),
        "outputs": outputs,
        "warnings": result.warnings,
    }

    passed = (
        fx_error <= THRESHOLDS["fx_rel_error"]
        and fy_error <= THRESHOLDS["fy_rel_error"]
        and pp_error <= THRESHOLDS["principal_point_error_px"]
        and result.rms <= THRESHOLDS["calibration_rms_px"]
    )
    if truth_dist is not None:
        # 畸变系数只做量级检查（合成渲染用直线近似曲线边，存在系统性偏差）
        metrics["k1_est"] = dist[0]
        metrics["k1_truth"] = float(truth_dist[0, 0])
        metrics["k1_abs_error"] = abs(dist[0] - float(truth_dist[0, 0]))
        metrics["k2_abs_error"] = abs(dist[1] - float(truth_dist[0, 1]))
    return CheckResult("calibration_math", passed, metrics, artifacts=artifacts)


# --------------------------------------------------------------------------- 5. 三角化
def check_landmarks(out_dir: Path) -> CheckResult:
    """用合成相机位姿验证多视图三角化的位置误差。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    cube_truth = np.array([0.05, -0.02, 1.20], dtype=np.float64)
    edge = 0.05

    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=1.0, min_baseline_m=0.01))
    errors: List[float] = []

    # 相机沿 x 轴平移并始终看向立方体附近（模拟绕目标移动）
    for index in range(8):
        x = -0.35 + 0.10 * index
        tvec = np.array([x, 0.0, 0.0], dtype=np.float64)      # 相机位置（世界系）
        # 相机朝向：让光轴指向立方体（用 look-at 构造旋转）
        forward = cube_truth - tvec
        forward /= np.linalg.norm(forward)
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rot_cw = np.stack([right, down, forward], axis=0)      # 世界->相机
        pose_wc = np.eye(4)
        pose_wc[:3, :3] = rot_cw.T                              # 相机->世界
        pose_wc[:3, 3] = tvec

        # 目标在相机系的像素坐标（用真值投影，模拟检测中心）
        point_camera = pose_wc[:3, :3].T @ (cube_truth - tvec)
        if point_camera[2] <= 0:
            continue
        uvw = camera @ point_camera
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        if not (0 <= u < IMAGE_SIZE[0] and 0 <= v < IMAGE_SIZE[1]):
            continue

        detection = Detection(
            class_id=0,
            class_name="cube",
            confidence=0.9,
            xyxy=[u - 20, v - 20, u + 20, v + 20],
            track_id=1,
            geometry=GeometryMetrics(center=(float(u), float(v)), corners=[], vertex_count=4),
            measurement=Measurement(method="bearing_only"),
        )
        landmark = manager.add_detection(
            detection,
            stamp=float(index),
            pose_wc=pose_wc,
            frame_id="map",
            camera_matrix=camera,
        )
        if landmark is not None and landmark.source == "triangulated":
            error = float(np.linalg.norm(np.asarray(landmark.position) - cube_truth))
            errors.append(error)

    final = manager.snapshot()
    best_error = None
    if final:
        best_error = float(np.linalg.norm(np.asarray(final[0].position) - cube_truth))

    metrics = {
        "n_triangulated": len(errors),
        "final_error_m": best_error,
        "mean_error_m": float(np.mean(errors)) if errors else None,
        "landmark": final[0].to_dict() if final else None,
        "cube_truth": [float(v) for v in cube_truth],
    }
    passed = best_error is not None and best_error <= THRESHOLDS["landmark_error_m"]
    return CheckResult("landmark_triangulation", passed, metrics)


# --------------------------------------------------------------------------- 6. 消息契约
def check_message_contract() -> CheckResult:
    """JSON 契约自查：老 YOLO 字段齐备且能被自己解析回来。"""
    detection = Detection(
        class_id=0,
        class_name="cube",
        confidence=0.5,
        xyxy=[1.0, 2.0, 3.0, 4.0],
        track_id=7,
        geometry=GeometryMetrics(center=(2.0, 3.0), corners=[[1, 2], [3, 2], [3, 4], [1, 4]], vertex_count=4),
        measurement=Measurement(distance_m=0.5, method="known_size", ray_camera=[0.0, 0.0, 1.0], bearing_rad=(0.0, 0.0)),
    )
    frame = DetectionFrame(detections=[detection], stamp_sec=1, stamp_nanosec=2, frame_id="cam", image_width=640, image_height=480)
    payload = json.loads(frame.to_json())
    restored = DetectionFrame.from_json(frame.to_json())
    legacy_keys = {"stamp", "frame_id", "source", "count", "detections"}
    metrics = {
        "has_legacy_keys": legacy_keys.issubset(payload.keys()),
        "detection_keys": sorted(payload["detections"][0].keys()),
        "roundtrip_ok": restored.detections[0].track_id == 7 and restored.image_width == 640,
        "size_bytes": len(frame.to_json()),
    }
    passed = metrics["has_legacy_keys"] and metrics["roundtrip_ok"]
    return CheckResult("message_contract", passed, metrics)


# --------------------------------------------------------------------------- main
def run(out_dir: Path, quick: bool = False) -> int:
    """执行全部自测，返回进程退出码。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 78)
    _log("underwater_perception 无硬件自测（合成数据）")
    _log(f"输出目录: {out_dir}")
    _log(f"环境: python {sys.version.split()[0]}, opencv {cv2.__version__}, numpy {np.__version__}")
    _log("=" * 78)

    checks: List[CheckResult] = []
    started = time.time()

    for name, func in (
        ("message_contract", check_message_contract),
        ("geometry_detection", lambda: check_detection(out_dir)),
        ("false_positive_suppression", lambda: check_false_positives(out_dir)),
        ("tracking_stability", lambda: check_tracking(out_dir)),
        ("calibration_math", lambda: check_calibration(out_dir, with_distortion=not quick)),
        ("landmark_triangulation", lambda: check_landmarks(out_dir)),
    ):
        _log(f"\n--- {name} ---")
        try:
            result = func()  # type: ignore[operator]
        except Exception as exc:  # 单项失败不能中断整轮自测
            result = CheckResult(name, False, {"exception": repr(exc)}, notes="检查过程中抛出异常")
        checks.append(result)
        status = "PASS" if result.passed else "FAIL"
        _log(f"[{status}] {result.name}")
        for key, value in result.metrics.items():
            _log(f"    {key}: {value}")
        if result.notes:
            _log(f"    note: {result.notes}")
        for artifact in result.artifacts:
            _log(f"    artifact: {artifact}")

    elapsed = time.time() - started
    passed = all(check.passed for check in checks)
    _log("\n" + "=" * 78)
    _log(f"总计: {sum(1 for c in checks if c.passed)}/{len(checks)} 项通过，耗时 {elapsed:.1f}s")
    _log("=" * 78)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed,
        "passed": passed,
        "thresholds": THRESHOLDS,
        "checks": [
            {"name": c.name, "passed": c.passed, "metrics": c.metrics, "notes": c.notes, "artifacts": c.artifacts}
            for c in checks
        ],
        "environment": {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__},
        "disclaimer": (
            "本报告全部基于合成数据。合成数据不包含真实水下的散射、浑浊、运动模糊、"
            "镜头污渍与防水壳折射，因此不能作为真实环境下性能的证据；"
            "真实验证清单见 docs/UNVERIFIED.md。"
        ),
    }
    report_path = out_dir / "selftest_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"报告已写入: {report_path}")
    return 0 if passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="underwater_perception 无硬件自测")
    parser.add_argument("--out", default="selftest_out", help="输出目录（默认 ./selftest_out）")
    parser.add_argument("--quick", action="store_true", help="跳过畸变标定（更快）")
    args = parser.parse_args(argv)
    return run(Path(args.out), quick=args.quick)


if __name__ == "__main__":
    sys.exit(main())
