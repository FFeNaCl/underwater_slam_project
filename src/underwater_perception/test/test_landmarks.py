"""语义地标（多视图三角化）的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from underwater_perception.core.landmarks import LandmarkManager
from underwater_perception.core.params import LandmarkConfig
from underwater_perception.core.types import Detection, GeometryMetrics, Measurement
from underwater_perception.core.synthetic import default_camera_matrix

IMAGE_SIZE = (1280, 480)
CUBE_TRUTH = np.array([0.05, -0.02, 1.20])


def _look_at_pose(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """构造"相机在 camera_position、看向 target"的 Twc（世界<-相机）。"""
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)  # 相机->世界
    pose[:3, 3] = camera_position
    return pose


def _detection_at(u: float, v: float, track_id: int = 1, distance=None) -> Detection:
    return Detection(
        class_id=0,
        class_name="cube",
        confidence=0.9,
        xyxy=[u - 15, v - 15, u + 15, v + 15],
        track_id=track_id,
        geometry=GeometryMetrics(center=(u, v), corners=[], vertex_count=6),
        measurement=Measurement(method="bearing_only", distance_m=distance),
    )


def _observe(manager: LandmarkManager, camera_position, noise_px: float = 0.0, seed: int = 0, track_id: int = 1):
    """在某相机位置生成一次观测并交给地标管理器。"""
    camera = default_camera_matrix(*IMAGE_SIZE)
    pose = _look_at_pose(np.asarray(camera_position, dtype=np.float64), CUBE_TRUTH)
    point_camera = pose[:3, :3].T @ (CUBE_TRUTH - pose[:3, 3])
    uvw = camera @ point_camera
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    if noise_px > 0.0:
        rng = np.random.default_rng(seed)
        u += float(rng.normal(0.0, noise_px))
        v += float(rng.normal(0.0, noise_px))
    return manager.add_detection(
        _detection_at(float(u), float(v), track_id=track_id),
        stamp=float(seed + 1),
        pose_wc=pose,
        frame_id="map",
        camera_matrix=camera,
    )


def test_triangulation_from_multiple_viewpoints() -> None:
    """绕目标移动时应三角化出接近真值的地标。"""
    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=1.0, min_baseline_m=0.01))
    landmark = None
    for index in range(8):
        landmark = _observe(manager, [-0.35 + 0.10 * index, 0.0, 0.0], seed=index)
    assert landmark is not None
    assert landmark.source == "triangulated"
    error = float(np.linalg.norm(landmark.position - CUBE_TRUTH))
    assert error < 1e-6, f"无噪声三角化误差过大: {error}"
    assert landmark.frame_id == "map"
    assert landmark.n_views >= 2


def test_triangulation_is_robust_to_pixel_noise() -> None:
    """加入 1 px 像素噪声后，地标误差应仍在厘米级（真实检测噪声量级）。"""
    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=1.0, min_baseline_m=0.01, max_reproj_error_px=20.0))
    landmark = None
    for index in range(10):
        landmark = _observe(manager, [-0.4 + 0.09 * index, 0.02 * (index % 3), 0.0], noise_px=1.0, seed=index)
    assert landmark is not None, "带噪声时未能三角化"
    error = float(np.linalg.norm(landmark.position - CUBE_TRUTH))
    assert error < 0.05, f"1 px 噪声下地标误差过大: {error:.4f} m"


def test_no_triangulation_without_baseline() -> None:
    """相机不动时不能三角化（视差为 0，解病态）。"""
    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=2.0, min_baseline_m=0.05, allow_scale_fallback=False))
    landmark = None
    for index in range(5):
        landmark = _observe(manager, [0.0, 0.0, 0.0], seed=index)
    assert landmark is None
    assert manager.snapshot() == []


def test_scale_fallback_when_no_pose() -> None:
    """没有 SLAM 位姿但有已知边长测距时，应退化为相机坐标系下的近似点。"""
    manager = LandmarkManager(LandmarkConfig(allow_scale_fallback=True, cube_edge_m=0.05))
    camera = default_camera_matrix(*IMAGE_SIZE)
    detection = _detection_at(camera[0, 2], camera[1, 2], distance=0.5)
    landmark = manager.add_detection(detection, stamp=1.0, pose_wc=None, frame_id="camera_optical", camera_matrix=camera)
    assert landmark is not None
    assert landmark.source == "scale_fallback"
    assert landmark.frame_id == "camera_optical"
    # 主点方向的射线是光轴，因此位置应约为 (0, 0, 0.5)
    assert np.allclose(landmark.position, [0.0, 0.0, 0.5], atol=1e-9)


def test_landmark_id_is_stable_for_same_track() -> None:
    """同一个 track_id 的多帧观测必须更新同一个地标，而不是每帧新建。"""
    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=0.5, min_baseline_m=0.005))
    ids = []
    for index in range(6):
        landmark = _observe(manager, [-0.2 + 0.08 * index, 0.0, 0.0], seed=index)
        if landmark is not None:
            ids.append(landmark.landmark_id)
    assert len(set(ids)) == 1, f"地标 ID 不稳定: {set(ids)}"
    assert len(manager.snapshot()) == 1


def test_prune_removes_stale_landmarks() -> None:
    """超时未更新的地标必须被清理（否则 RViz 里会残留幽灵目标）。"""
    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=0.5, min_baseline_m=0.005, stale_after_s=1.0))
    landmark = None
    for index in range(4):
        landmark = _observe(manager, [-0.2 + 0.08 * index, 0.0, 0.0], seed=index)
    assert landmark is not None and len(manager.snapshot()) == 1
    removed = manager.prune(now=landmark.last_stamp + 10.0)
    assert removed == [landmark.landmark_id]
    assert manager.snapshot() == []


def test_landmark_serialises_to_message_contract() -> None:
    """地标的 JSON 字典必须含下游需要的字段。"""
    manager = LandmarkManager(LandmarkConfig(min_views=2, min_parallax_deg=0.5, min_baseline_m=0.005))
    landmark = None
    for index in range(4):
        landmark = _observe(manager, [-0.2 + 0.08 * index, 0.0, 0.0], seed=index)
    assert landmark is not None
    payload = landmark.to_dict()
    for key in ("landmark_id", "track_id", "class_name", "frame_id", "position", "n_views", "source", "confidence"):
        assert key in payload, f"地标消息缺字段: {key}"
    assert len(payload["position"]) == 3
