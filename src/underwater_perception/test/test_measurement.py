"""单目测量与位姿拟合的单元测试。

包含两个**回归测试**，对应开发过程中真实出现过的 bug（这类 bug 不会报错，只会静默给出错误结果）：
1. ``test_face_pose_ordering_regression``：solvePnP(IPPE_SQUARE) 的角点顺序不匹配时
   OpenCV 不报错而是返回错误位姿（实测 0.005 m vs 真值 0.45 m）。
2. ``test_silhouette_residual_length_is_fixed``：投影轮廓顶点数会随位姿变化，
   若残差向量长度随之变化，代价函数不可比，优化器会一步都不动。
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from underwater_perception.core.measurement import (
    _silhouette_residuals,
    bearing_from_ray,
    distance_from_known_size,
    face_pose_from_quad,
    invert_pose,
    matrix_to_pose,
    pixel_to_ray,
    pose_to_matrix,
    refine_cube_pose,
    triangulate_rays,
)
from underwater_perception.core.synthetic import (
    cube_object_points,
    default_camera_matrix,
    render_cube_scene,
    render_underwater_background,
)

IMAGE_SIZE = (1280, 480)
CUBE_EDGE = 0.05


def test_pixel_to_ray_at_principal_point_is_optical_axis() -> None:
    """主点对应的射线必须就是光轴 (0,0,1)。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    ray = pixel_to_ray(camera[0, 2], camera[1, 2], camera)
    assert np.allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)
    azimuth, elevation = bearing_from_ray(ray)
    assert azimuth == pytest.approx(0.0, abs=1e-9)
    assert elevation == pytest.approx(0.0, abs=1e-9)
    # 图像右侧的点方位角为正，下方的点俯仰角为正（y 向下）
    right = pixel_to_ray(camera[0, 2] + 100, camera[1, 2], camera)
    down = pixel_to_ray(camera[0, 2], camera[1, 2] + 100, camera)
    assert bearing_from_ray(right)[0] > 0
    assert bearing_from_ray(down)[1] > 0


def test_known_size_distance_formula() -> None:
    """Z = f·L/s 的基本正确性。"""
    assert distance_from_known_size(70.0, 700.0, 0.05) == pytest.approx(0.5)
    assert distance_from_known_size(0.0, 700.0, 0.05) is None
    assert distance_from_known_size(70.0, 700.0, 0.0) is None


def test_face_pose_ordering_regression() -> None:
    """回归测试：角点顺序任意时，face_pose_from_quad 仍必须给出正确距离。

    做法：把真实投影出的正方形角点做**循环移位与反向**，模拟 approxPolyDP 的任意顺序，
    要求所有 8 种顺序都恢复出接近真值的距离。
    """
    camera = default_camera_matrix(*IMAGE_SIZE)
    truth_distance = 0.45
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [truth_distance]])
    points = cube_object_points(CUBE_EDGE)
    front_face = points[[0, 1, 2, 3]]  # z = -h 面
    projected, _ = cv2.projectPoints(front_face, rvec, tvec, camera, np.zeros((5, 1)))
    corners = projected.reshape(4, 2)

    for shift in range(4):
        for direction in (1, -1):
            permuted = np.roll(corners[::direction], shift, axis=0)
            ok, _, tvec_est, _, error = face_pose_from_quad(permuted, CUBE_EDGE, camera, None)
            assert ok, f"顺序 shift={shift} direction={direction} 求解失败"
            # face_pose_from_quad 的物体原点在**可见面中心**，因此恢复出的距离是
            # 面中心到相机光心的距离 = 0.45 - h（h = 半边长 = 0.025），即 0.425 m。
            expected_face_center = truth_distance - CUBE_EDGE / 2.0
            distance = float(np.linalg.norm(np.asarray(tvec_est).reshape(3)))
            assert distance == pytest.approx(expected_face_center, rel=0.02), (
                f"角点顺序 shift={shift} direction={direction} 得到错误距离 {distance:.4f}"
            )
            assert error < 1.0


def test_silhouette_residual_length_is_fixed() -> None:
    """回归测试：残差向量长度必须固定，不随投影轮廓顶点数变化。"""
    from underwater_perception.core.measurement import SILHOUETTE_SAMPLES

    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    hexagon = np.array(
        [[50.0, 0.0], [100.0, 25.0], [100.0, 75.0], [50.0, 100.0], [0.0, 75.0], [0.0, 25.0]]
    )
    assert len(_silhouette_residuals(square, square)) == 2 * SILHOUETTE_SAMPLES
    assert len(_silhouette_residuals(square, hexagon)) == 2 * SILHOUETTE_SAMPLES
    assert len(_silhouette_residuals(hexagon, square)) == 2 * SILHOUETTE_SAMPLES


def test_refine_cube_pose_recovers_distance_for_oblique_view() -> None:
    """斜视（六边形轮廓）时轮廓拟合必须给出准确距离——这正是引入它的原因。

    对比：正视假设在斜视时有 ~25% 系统误差（见 selftest 报告），拟合后应 <5%。
    """
    camera = default_camera_matrix(*IMAGE_SIZE)
    background = render_underwater_background(*IMAGE_SIZE, seed=21)
    for rvec, tvec in (((0.45, 0.55, 0.20), (0.0, 0.0, 0.45)), ((-0.35, 0.40, -0.25), (0.03, -0.02, 0.55))):
        scene = render_cube_scene(
            camera, rvec=rvec, tvec=tvec, edge_m=CUBE_EDGE, image_size=IMAGE_SIZE, background=background
        )
        truth = float(np.linalg.norm(np.asarray(tvec)))
        ok, _, tvec_fit, rms = refine_cube_pose(scene.silhouette.reshape(-1, 2), CUBE_EDGE, camera, None)
        assert ok, f"轮廓拟合失败: {rvec}"
        distance = float(np.linalg.norm(np.asarray(tvec_fit).reshape(3)))
        assert abs(distance - truth) / truth < 0.05, f"距离误差过大: {distance:.3f} vs {truth:.3f}"
        assert rms < 5.0


def test_measurement_distance_refers_to_cube_center() -> None:
    """回归测试：测量出的距离必须是**立方体中心**的距离，不是可见面中心。

    背景：``face_pose_from_quad`` 的物体原点在可见面上，直接取 ``|tvec|`` 会比立方体中心
    近一个半边长（5 cm 立方体 = 2.5 cm，约 5.6% 系统偏差）。开发过程中自测的
    ``max_distance_rel_error`` 从 0.9% 跳到 5.9% 才暴露出来。两条路径（4 角点 PnP 与
    轮廓拟合）的 ``pose_tvec`` 语义必须一致。
    """
    from underwater_perception.core.cube_geometry import detect_observations
    from underwater_perception.core.params import CubeDetectorConfig
    from underwater_perception.core.synthetic import render_underwater_background

    camera = default_camera_matrix(*IMAGE_SIZE)
    background = render_underwater_background(*IMAGE_SIZE, seed=31)
    truth_distance = 0.45
    flat = CubeDetectorConfig().to_flat()
    flat.update(
        {
            "camera_fx": float(camera[0, 0]),
            "camera_fy": float(camera[1, 1]),
            "camera_cx": float(camera[0, 2]),
            "camera_cy": float(camera[1, 2]),
            "cube_edge_m": CUBE_EDGE,
        }
    )
    config = CubeDetectorConfig.from_flat(flat)

    # 正视：会走 4 角点 PnP 路径；斜视：走轮廓拟合路径。两条都必须指向立方体中心。
    for rvec in ((0.0, 0.0, 0.0), (0.45, 0.55, 0.20)):
        scene = render_cube_scene(
            camera, rvec=rvec, tvec=(0.0, 0.0, truth_distance), edge_m=CUBE_EDGE,
            image_size=IMAGE_SIZE, background=background,
        )
        observations = detect_observations(scene.image, config)
        assert observations, f"未检出: rvec={rvec}"
        measurement = observations[0].detection.measurement
        assert measurement is not None and measurement.distance_m is not None
        relative_error = abs(measurement.distance_m - truth_distance) / truth_distance
        assert relative_error < 0.03, (
            f"rvec={rvec} 方法={measurement.method} 距离={measurement.distance_m:.4f} "
            f"与真值 {truth_distance} 相差 {relative_error * 100:.1f}%"
        )
        # pose_tvec 也必须是立方体中心
        assert measurement.pose_tvec is not None
        assert float(np.linalg.norm(measurement.pose_tvec)) == pytest.approx(truth_distance, rel=0.03)


def test_triangulation_is_exact_without_noise() -> None:
    """无噪声时三角化应精确恢复三维点（验证最小二乘推导正确）。"""
    truth = np.array([0.3, -0.2, 1.7])
    poses = [
        np.array([[1.0, 0.0, 0.0, -0.4], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0, 0.4], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ]
    origins, directions = [], []
    for pose in poses:
        point_camera = pose[:3, :3].T @ (truth - pose[:3, 3])
        origins.append(pose[:3, 3])
        directions.append(point_camera / np.linalg.norm(point_camera))
    point, rms, n_views = triangulate_rays(origins, directions)
    assert point is not None
    assert n_views == 3
    assert np.linalg.norm(point - truth) < 1e-9
    assert rms < 1e-9


def test_triangulation_rejects_parallel_rays() -> None:
    """近平行射线（无基线）必须拒绝，而不是给出一个荒谬的点。"""
    origins = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0001]]
    directions = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    point, rms, _ = triangulate_rays(origins, directions)
    assert point is None or not np.isfinite(rms)
    # 单条射线不构成三角化
    point_one, _, _ = triangulate_rays([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
    assert point_one is None


def test_pose_matrix_roundtrip_and_inverse() -> None:
    """位姿矩阵与四元数互转、以及刚体求逆的正确性。"""
    position = [0.3, -0.2, 1.1]
    quaternion = [0.1, 0.2, 0.3, math.sqrt(1.0 - 0.01 - 0.04 - 0.09)]
    matrix = pose_to_matrix(position, quaternion)
    position_back, quaternion_back = matrix_to_pose(matrix)
    assert position_back == pytest.approx(position)
    for a, b in zip(quaternion, quaternion_back):
        assert a == pytest.approx(b, abs=1e-9)

    inverse = invert_pose(matrix)
    assert np.allclose(matrix @ inverse, np.eye(4), atol=1e-12)
