"""标定数学与输出文件的单元测试（用合成棋盘格，无需相机）。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from underwater_perception.core.calibration import (
    calibrate,
    collect_views,
    detect_pattern,
    undistort_image,
    view_from_array,
    write_all_outputs,
    write_enhancer_yaml,
    write_orbslam3_yaml,
    write_ost_yaml,
)
from underwater_perception.core.params import CalibrationConfig
from underwater_perception.core.synthetic import (
    default_camera_matrix,
    make_chessboard_view_set,
    render_chessboard_scene,
    render_underwater_background,
)

IMAGE_SIZE = (1280, 480)


def _views(config: CalibrationConfig, n_views: int = 16, with_distortion: bool = True):
    """生成一组合成棋盘格视图。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    dist = np.array([[0.05, -0.08, 0.001, -0.0008, 0.01]]) if with_distortion else None
    scenes = make_chessboard_view_set(
        camera, n_views=n_views, image_size=IMAGE_SIZE, dist_coeffs=dist, seed=5
    )
    accepted, rejected = collect_views([scene.image for scene in scenes], config)
    return camera, dist, accepted, rejected


def test_detect_pattern_finds_all_inner_corners() -> None:
    """渲染的棋盘格上必须检出全部 8x6=48 个内角点，且位置与真值点阵一致。

    注意：这里比较的是**点集**而不是逐点顺序。OpenCV 返回的角点顺序取决于它锁定的
    起始角点与遍历方向（可能与我们的行优先顺序差一个旋转/转置），这对标定没有影响
    （平面靶标的法向不可观测，转置等价于把板子翻面，能被每帧位姿吸收），
    但逐点相减会得出"偏差上百像素"的错误结论——开发过程中就因此误判过一次。
    """
    camera = default_camera_matrix(*IMAGE_SIZE)
    config = CalibrationConfig()
    background = render_underwater_background(*IMAGE_SIZE, seed=3)
    scene = render_chessboard_scene(
        camera, rvec=(0.2, 0.25, 0.1), tvec=(0.0, 0.0, 0.55), image_size=IMAGE_SIZE, background=background
    )
    corners, method = detect_pattern(scene.image, config)
    assert corners is not None, f"未检出棋盘格 (method={method})"
    assert len(corners) == config.cols * config.rows

    detected = corners.reshape(-1, 2).astype(np.float64)
    truth = scene.corners.reshape(-1, 2).astype(np.float64)

    # 集合比较：每个检出角点到最近真值角点的距离
    distances = np.array([np.min(np.linalg.norm(truth - point, axis=1)) for point in detected])
    assert float(np.mean(distances)) < 2.0, f"检出角点与真值点阵不符: 平均 {np.mean(distances):.2f} px"
    assert float(np.max(distances)) < 4.0, f"存在明显偏离的角点: 最大 {np.max(distances):.2f} px"

    # 点阵形状校验：两个方向上的角点跨度比例应与 8:6 一致（能发现行列互换）
    span_detected = detected.max(axis=0) - detected.min(axis=0)
    span_truth = truth.max(axis=0) - truth.min(axis=0)
    assert span_detected[0] == pytest.approx(span_truth[0], rel=0.25)
    assert span_detected[1] == pytest.approx(span_truth[1], rel=0.25)


def test_view_without_board_is_rejected() -> None:
    """没有棋盘的图像必须被拒绝，而不是产生一个"零角点视图"污染标定。"""
    config = CalibrationConfig()
    background = render_underwater_background(*IMAGE_SIZE, seed=9)
    result = view_from_array(background, config)
    assert result.found is False
    assert result.accepted is False
    assert result.reason == "pattern_not_found"


def test_calibration_recovers_intrinsics() -> None:
    """标定必须恢复出接近真值的内参（这是标定程序的核心验收项）。"""
    config = CalibrationConfig(min_views=12)
    camera, _, accepted, _ = _views(config)
    assert len(accepted) >= 12, f"有效视图不足: {len(accepted)}"

    result = calibrate(accepted, IMAGE_SIZE, config, source="synthetic")
    assert result.ok, result.message
    assert abs(result.fx - camera[0, 0]) / camera[0, 0] < 0.03
    assert abs(result.fy - camera[1, 1]) / camera[1, 1] < 0.03
    assert math.hypot(result.cx - camera[0, 2], result.cy - camera[1, 2]) < 8.0
    assert result.rms < 1.0
    assert result.quality_grade() in ("good", "fair")
    assert len(result.per_view_errors) == result.n_views_used


def test_calibration_reports_small_sample_warning() -> None:
    """视图过少时必须给出明确警告（现场最常见的失败原因）。"""
    config = CalibrationConfig(min_views=12)
    camera = default_camera_matrix(*IMAGE_SIZE)
    scenes = make_chessboard_view_set(camera, n_views=5, image_size=IMAGE_SIZE, seed=11)
    accepted, _ = collect_views([scene.image for scene in scenes], config)
    result = calibrate(accepted, IMAGE_SIZE, config)
    assert result.ok
    assert any("少于建议值" in w for w in result.warnings)


def test_calibration_fails_loudly_with_too_few_views() -> None:
    """视图不足 3 个时必须明确失败，而不是返回一个乱数内参。"""
    config = CalibrationConfig()
    camera = default_camera_matrix(*IMAGE_SIZE)
    scenes = make_chessboard_view_set(camera, n_views=2, image_size=IMAGE_SIZE, seed=13)
    accepted, _ = collect_views([scene.image for scene in scenes], config)
    result = calibrate(accepted, IMAGE_SIZE, config)
    assert result.ok is False
    assert result.message == "not_enough_views"
    assert result.camera_matrix is None


def test_output_files_are_parseable_and_consistent(tmp_path: Path) -> None:
    """四件套输出必须能被解析，且数值与标定结果一致（下游全靠这些文件）。"""
    config = CalibrationConfig(min_views=12, camera_name="water_test", orbslam3_settings_name="water_1280x480")
    camera, _, accepted, _ = _views(config)
    result = calibrate(accepted, IMAGE_SIZE, config)
    outputs = write_all_outputs(tmp_path, result, config, views=accepted)

    # ost.yaml：ROS camera_calibration 兼容格式
    ost = yaml.safe_load(Path(outputs["ost"]).read_text(encoding="utf-8"))
    assert ost["image_width"] == IMAGE_SIZE[0]
    assert ost["image_height"] == IMAGE_SIZE[1]
    assert ost["camera_name"] == "water_test"
    assert ost["distortion_model"] == "plumb_bob"
    assert len(ost["camera_matrix"]["data"]) == 9
    assert len(ost["distortion_coefficients"]["data"]) == 5
    assert ost["camera_matrix"]["data"][0] == pytest.approx(result.fx)
    assert ost["projection_matrix"]["rows"] == 3
    assert ost["avg_reprojection_error"] == pytest.approx(result.rms)

    # ORB-SLAM3 设置文件：键名必须与 ORB-SLAM3 期望完全一致
    orbslam3_text = Path(outputs["orbslam3"]).read_text(encoding="utf-8")
    assert orbslam3_text.startswith("%YAML:1.0")
    for key in (
        "Camera.fx:", "Camera.fy:", "Camera.cx:", "Camera.cy:",
        "Camera.k1:", "Camera.k2:", "Camera.p1:", "Camera.p2:", "Camera.k3:",
        "Camera.width:", "Camera.height:", "Camera.fps:", "Camera.RGB:",
        "ORBextractor.nFeatures:", "ORBextractor.iniThFAST:", "Viewer.PointSize:",
    ):
        assert key in orbslam3_text, f"缺少年 ORB-SLAM3 键: {key}"
    assert f"Camera.fx: {result.fx:.6f}" in orbslam3_text
    assert 'Camera.type: "PinHole"' in orbslam3_text

    # 增强节点参数：camera_matrix 必须 9 个、distortion_coeffs 5 个
    enhancer = yaml.safe_load(Path(outputs["enhancer"]).read_text(encoding="utf-8"))
    params = enhancer["image_enhancer_cpp"]["ros__parameters"]
    assert params["use_undistort"] is True
    assert len(params["camera_matrix"]) == 9
    assert len(params["distortion_coeffs"]) == 5
    assert params["camera_matrix"][0] == pytest.approx(result.fx)

    # 报告 JSON
    import json

    report = json.loads(Path(outputs["report"]).read_text(encoding="utf-8"))
    assert report["result"]["ok"] is True
    assert report["result"]["views_used"] == result.n_views_used
    assert report["environment"]["opencv"]
    assert report["result"]["pinhole_approx_note"]


def test_individual_writers(tmp_path: Path) -> None:
    """单独写文件时也必须自动创建父目录（CLI 会写嵌套路径）。"""
    config = CalibrationConfig(min_views=12)
    _, _, accepted, _ = _views(config)
    result = calibrate(accepted, IMAGE_SIZE, config)
    nested = tmp_path / "a" / "b"
    assert Path(write_ost_yaml(nested / "ost.yaml", result, config)).is_file()
    assert Path(write_orbslam3_yaml(nested / "slam.yaml", result, config)).is_file()
    assert Path(write_enhancer_yaml(nested / "enh.yaml", result, config)).is_file()


def test_undistort_image_keeps_size() -> None:
    """去畸变输出必须与输入同尺寸（下游按同尺寸处理像素坐标）。"""
    config = CalibrationConfig(min_views=12)
    _, _, accepted, _ = _views(config)
    result = calibrate(accepted, IMAGE_SIZE, config)
    sample = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    undistorted, new_matrix, roi = undistort_image(sample, result, alpha=0.0)
    assert undistorted.shape == sample.shape
    assert new_matrix.shape == (3, 3)
    assert len(roi) == 4
