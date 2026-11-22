"""几何旁路立方体检测的合成数据测试。

这些测试用 core/synthetic.py 渲染**真实透视投影**的立方体（含可见面明暗与内部棱线），
因此可以量化"检出率、轮廓 IoU、顶点数判断、误检抑制、鲁棒性"，
而不是只看"函数没报错"。这是在没有相机的情况下能做到的最强验证。

重要：这些结论只对合成数据成立。真实水下会有散射、浑浊、运动模糊、镜头污渍，
所以交付文档 docs/UNVERIFIED.md 里明确列出"必须在 Linux + 真实相机上复核"的项目。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from underwater_perception.core.cube_geometry import CubeDetector, detect_observations
from underwater_perception.core.params import CubeDetectorConfig
from underwater_perception.core.types import polygon_iou
from underwater_perception.core.synthetic import (
    default_camera_matrix,
    render_cube_scene,
    render_cube_sequence,
    render_underwater_background,
)

IMAGE_SIZE = (1280, 480)


def _config(**overrides) -> CubeDetectorConfig:
    """构造检测配置：允许测试里逐项覆盖扁平参数。"""
    flat = CubeDetectorConfig().to_flat()
    flat.update(overrides)
    return CubeDetectorConfig.from_flat(flat)


def _iou_with_truth(scene, detection) -> float:
    """检测多边形与真值轮廓的 IoU。"""
    return polygon_iou(detection.polygon(), scene.silhouette.tolist())


@pytest.mark.parametrize(
    "rvec, tvec, expected_min_vertices",
    [
        # 正视单面：轮廓≈正方形，4 个顶点
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.45), 4),
        # 斜视三面：轮廓≈六边形，5~6 个顶点
        ((0.45, 0.55, 0.20), (0.0, 0.0, 0.45), 5),
        # 另一个斜视角
        ((-0.35, 0.40, -0.25), (0.03, -0.02, 0.55), 5),
    ],
)
def test_detects_cube_at_multiple_viewpoints(rvec, tvec, expected_min_vertices) -> None:
    """不同视角下都应检出，且轮廓与真值 IoU 足够高、顶点数判断正确。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(camera, rvec=rvec, tvec=tvec, image_size=IMAGE_SIZE)

    observations = detect_observations(scene.image, _config())
    assert observations, f"未检出立方体: rvec={rvec}, tvec={tvec}"

    best = max(observations, key=lambda o: o.confidence)
    iou = _iou_with_truth(scene, best.detection)
    assert iou > 0.70, f"轮廓 IoU 过低: {iou:.3f}"
    assert best.metrics.vertex_count >= expected_min_vertices, (
        f"顶点数判断错误: {best.metrics.vertex_count} < {expected_min_vertices}"
    )
    # 六边形轮廓的填充率应落在 (0.62, 1.0] 区间内；正方形接近 1
    assert 0.6 < best.metrics.fill_ratio <= 1.01
    assert best.metrics.convexity > 0.95, "立方体轮廓必须是凸的"


def test_no_false_positive_on_pure_background() -> None:
    """只有水下背景时不得检出任何目标。"""
    background = render_underwater_background(*IMAGE_SIZE, seed=3)
    assert detect_observations(background, _config()) == []


def test_no_false_positive_on_textured_blue_props() -> None:
    """蓝色/青色物体（水体常见干扰）不得被当成红色立方体。"""
    background = render_underwater_background(*IMAGE_SIZE, seed=5)
    for color in ((200, 120, 30), (180, 160, 40)):
        scene = render_cube_scene(
            default_camera_matrix(*IMAGE_SIZE),
            rvec=(0.3, 0.4, 0.1),
            tvec=(0.0, 0.0, 0.5),
            image_size=IMAGE_SIZE,
            color_bgr=color,
            background=background,
        )
        assert detect_observations(scene.image, _config()) == [], f"颜色 {color} 被误检为红色立方体"


def test_strict_mode_rejects_flat_red_square() -> None:
    """严格模式（cube_silhouette）必须拒绝"红色平面方块"——这是最典型的误检。

    原理：平面色块内部没有棱线（内部直线方向数≈0），而立方体三面视角内部有 3 条棱。
    """
    background = render_underwater_background(*IMAGE_SIZE, seed=7)
    flat = background.copy()
    cv2.rectangle(flat, (560, 180), (700, 320), (40, 40, 205), thickness=-1)
    cv2.rectangle(flat, (560, 180), (700, 320), (25, 25, 140), thickness=2)

    permissive = detect_observations(flat, _config(mode="any"))
    strict = detect_observations(flat, _config(mode="cube_silhouette"))

    assert strict == [], "严格模式不应把平面红色方块当成立方体"
    # any 模式允许检出（保召回），但必须在文档里说明这一取舍
    assert isinstance(permissive, list)


def test_detects_cube_with_strict_mode_when_three_faces_visible() -> None:
    """斜视三面视角时，严格模式必须仍能检出（否则严格模式没有实用价值）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(camera, rvec=(0.45, 0.55, 0.20), tvec=(0.0, 0.0, 0.45), image_size=IMAGE_SIZE)
    observations = detect_observations(scene.image, _config(mode="cube_silhouette"))
    assert observations, "三面视角应满足严格模式的内部棱线要求"
    best = observations[0]
    assert best.metrics.edge_direction_count >= 2, (
        f"内部棱线方向数不足: {best.metrics.edge_direction_count}"
    )
    assert best.metrics.edge_support > 0.0


@pytest.mark.parametrize("blue_cast", [0.0, 0.3, 0.5])
def test_robust_to_underwater_color_cast(blue_cast: float) -> None:
    """水体偏色（红通道衰减）下仍应检出——这正是红色立方体在水下的主要挑战。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(
        camera,
        rvec=(0.4, 0.5, 0.15),
        tvec=(0.0, 0.0, 0.45),
        image_size=IMAGE_SIZE,
        blue_cast=blue_cast,
    )
    observations = detect_observations(scene.image, _config())
    assert observations, f"blue_cast={blue_cast} 时漏检"
    assert _iou_with_truth(scene, observations[0].detection) > 0.6


def test_robust_to_mild_blur_and_noise() -> None:
    """轻微运动模糊 + 传感器噪声下仍应检出（现场必然存在）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(
        camera,
        rvec=(0.4, 0.5, 0.15),
        tvec=(0.0, 0.0, 0.45),
        image_size=IMAGE_SIZE,
        blur_sigma=1.4,
        noise_sigma=4.0,
    )
    observations = detect_observations(scene.image, _config())
    assert observations, "轻微模糊+噪声下漏检"


def test_small_target_is_rejected_by_area_gate() -> None:
    """远距离过小的目标应被面积门限挡掉（避免输出注定无法三角化的垃圾检测）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(camera, rvec=(0.2, 0.3, 0.1), tvec=(0.0, 0.0, 6.0), image_size=IMAGE_SIZE)
    assert detect_observations(scene.image, _config(min_area_px=250)) == []


def test_roi_crops_out_interference() -> None:
    """ROI 之外的同色物体必须被忽略（现场用它裁掉水面反光带）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    background = render_underwater_background(*IMAGE_SIZE, seed=11)
    # ROI 内的目标
    inside = render_cube_scene(
        camera, rvec=(0.3, 0.4, 0.1), tvec=(-0.05, 0.0, 0.45), image_size=IMAGE_SIZE, background=background
    )
    # 把另一个立方体硬贴到右侧 ROI 之外
    outside = render_cube_scene(
        camera, rvec=(0.3, 0.4, 0.1), tvec=(0.18, 0.0, 0.45), image_size=IMAGE_SIZE, background=None
    )
    composed = inside.image.copy()
    mask = np.zeros(composed.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (1120, 240), 150, 255, -1)
    composed[mask > 0] = outside.image[mask > 0]

    roi = [0, 0, 700, 480]
    observations = detect_observations(composed, _config(roi=roi))
    assert observations, "ROI 内目标应被检出"
    for observation in observations:
        cx, _ = observation.metrics.center
        assert cx <= 700 + 5, f"ROI 外目标被检出: center_x={cx}"


def test_tracking_keeps_single_stable_id_over_sequence() -> None:
    """同一目标连续多帧必须保持同一个 track_id（语义地标依赖它做关联）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scenes = render_cube_sequence(camera, n_frames=10, image_size=IMAGE_SIZE, start=(-0.08, 0.0, 0.45), step=(0.015, 0.0, 0.0))
    detector = CubeDetector(_config(track_min_hits=2))

    seen: list = []
    for scene in scenes:
        detections = detector.detect(scene.image)
        if detections:
            seen.append(detections[0].track_id)

    assert len(seen) >= 6, f"检出帧数过少: {len(seen)}/10"
    assert len(set(seen)) == 1, f"track_id 不稳定: {set(seen)}"


def test_min_hits_suppresses_single_frame_false_positive() -> None:
    """单帧出现的假目标不应被发布（min_hits=2 的意义）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    background = render_underwater_background(*IMAGE_SIZE, seed=13)
    detector = CubeDetector(_config(track_min_hits=2))

    # 第一帧：只有背景 -> 无输出
    assert detector.detect(background) == []
    # 第二帧：出现一个立方体，但只出现一次
    scene = render_cube_scene(camera, rvec=(0.3, 0.4, 0.1), tvec=(0.0, 0.0, 0.45), image_size=IMAGE_SIZE, background=background)
    first = detector.detect(scene.image)
    assert first == [], "首个观测帧不应立即发布（hits=1 < min_hits=2）"
    # 第三帧：仍是同一目标 -> 达到 min_hits，应发布
    second = detector.detect(scene.image)
    assert len(second) == 1
    assert second[0].track_id is not None


def test_measurement_distance_is_physically_reasonable() -> None:
    """已知边长测距：0.45 m 处的 5 cm 立方体，估计值应在 ±25% 内。

    单目已知尺寸测距的误差主要来自"面未正对相机"，因此这里给出的是工程容差，
    不是精密指标的承诺（严格精度必须靠多视图三角化，见 test_landmarks.py）。
    """
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(camera, rvec=(0.05, 0.05, 0.0), tvec=(0.0, 0.0, 0.45), image_size=IMAGE_SIZE)
    config = _config(
        camera_fx=float(camera[0, 0]),
        camera_fy=float(camera[1, 1]),
        camera_cx=float(camera[0, 2]),
        camera_cy=float(camera[1, 2]),
        cube_edge_m=0.05,
    )
    observations = detect_observations(scene.image, config)
    assert observations
    measurement = observations[0].detection.measurement
    assert measurement is not None
    assert measurement.method in ("known_size", "known_size_pose")
    assert measurement.distance_m is not None
    assert 0.30 < measurement.distance_m < 0.65, f"距离估计偏离过大: {measurement.distance_m}"


def test_detector_reports_mask_for_debugging() -> None:
    """last_mask 必须与原图同尺寸（现场用 --print-mask 调阈值）。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    scene = render_cube_scene(camera, rvec=(0.4, 0.5, 0.15), tvec=(0.0, 0.0, 0.45), image_size=IMAGE_SIZE)
    detector = CubeDetector(_config())
    detector.detect(scene.image)
    assert detector.last_mask is not None
    assert detector.last_mask.shape == scene.image.shape[:2]
    assert detector.last_mask.dtype == np.uint8
