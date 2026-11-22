"""语义地标：把多帧检测结果三角化成地图里的三维点。

思路（这是"识别 + 建图"真正结合的地方）：

1. 几何旁路给出目标的**像素轮廓**与 ``track_id``；
2. SLAM 给出每一帧的**相机位姿** ``T_wc``（世界<-相机，见 docs/SLAM_INTEGRATION.md）；
3. 由像素中心得到相机坐标系射线 ``d_c``，旋转到世界系 ``d_w = R_wc · d_c``，起点为 ``t_wc``；
4. 同一 ``track_id`` 的多条射线用**线性最小二乘**求交点 -> 目标在地图中的三维位置；
5. 用重投影误差剔除坏观测，输出 ``MarkerArray`` 供 RViz 显示。

为什么必须三角化而不是"用像素大小算距离"：单目 SLAM 的地图尺度由初始化决定（不是米），
而三角化得到的地标与地图点云**共享同一尺度**，因此能和点云、轨迹画在同一张图里；
反之用已知边长算出的"米"直接塞进地图会尺度不一致。

当 SLAM 尚未初始化（没有位姿）时，退化为 ``scale_fallback``：用已知立方体边长估计距离，
得到**相机坐标系**下的点，并以检测消息的 ``frame_id`` 作为参考系发布。这能保证
"SLAM 一启动就能看到东西"，但绝不声称它和地图同尺度。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .measurement import invert_pose, pixel_to_ray, triangulate_rays
from .params import LandmarkConfig
from .types import Detection

__all__ = ["Observation", "Landmark", "LandmarkManager"]


@dataclass
class Observation:
    """一次观测：某帧里某个 track_id 的射线。"""

    track_id: int
    stamp: float
    ray_camera: np.ndarray                 # 相机坐标系单位射线
    pose_wc: Optional[np.ndarray] = None   # (4,4) 世界<-相机；None 表示该帧没有位姿
    distance_m: Optional[float] = None
    confidence: float = 0.0
    class_name: str = "cube"
    center_px: Tuple[float, float] = (0.0, 0.0)
    image_size: Tuple[int, int] = (0, 0)
    frame_id: str = ""

    @property
    def origin_world(self) -> Optional[np.ndarray]:
        """射线起点（相机光心）在世界系下的位置。"""
        if self.pose_wc is None:
            return None
        return np.asarray(self.pose_wc, dtype=np.float64)[:3, 3].copy()

    @property
    def direction_world(self) -> Optional[np.ndarray]:
        """射线方向在世界系下的单位向量。"""
        if self.pose_wc is None:
            return None
        rot = np.asarray(self.pose_wc, dtype=np.float64)[:3, :3]
        direction = rot @ self.ray_camera.reshape(3)
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm > 1e-12 else direction


@dataclass
class Landmark:
    """一个语义地标（地图中的三维目标）。"""

    landmark_id: int
    track_id: int
    position: np.ndarray
    frame_id: str
    class_name: str = "cube"
    n_views: int = 0
    reproj_error_px: Optional[float] = None
    triangulation_error: Optional[float] = None
    source: str = "triangulated"          # triangulated | scale_fallback
    confidence: float = 0.0
    first_stamp: float = 0.0
    last_stamp: float = 0.0
    distance_m: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        """消息契约：与 ``/semantic/landmarks`` 的 JSON 字段一一对应。"""
        return {
            "landmark_id": int(self.landmark_id),
            "track_id": int(self.track_id),
            "class_name": str(self.class_name),
            "frame_id": str(self.frame_id),
            "position": [float(v) for v in np.asarray(self.position).reshape(3)],
            "n_views": int(self.n_views),
            "reproj_error_px": None if self.reproj_error_px is None else float(self.reproj_error_px),
            "triangulation_error": None if self.triangulation_error is None else float(self.triangulation_error),
            "source": str(self.source),
            "confidence": float(self.confidence),
            "first_stamp": float(self.first_stamp),
            "last_stamp": float(self.last_stamp),
            "distance_m": None if self.distance_m is None else float(self.distance_m),
        }


class LandmarkManager:
    """按 ``track_id`` 累积观测并三角化出地标。

    Args:
        config: 地标参数（最少视图数、视差、误差门限等）。

    Notes:
        线程安全：ROS 节点里位姿与检测来自不同回调。请在节点侧用同一把锁保护
        ``add_detection`` / ``set_last_pose`` / ``snapshot`` 的调用（本类自身不加锁，
        以保持纯函数式的可测试性）。
    """

    def __init__(self, config: Optional[LandmarkConfig] = None) -> None:
        self.config = config or LandmarkConfig()
        self._observations: Dict[int, List[Observation]] = {}
        self._landmarks: Dict[int, Landmark] = {}
        self._next_landmark_id = 1
        self._track_to_landmark: Dict[int, int] = {}
        self.last_pose: Optional[np.ndarray] = None
        self.last_pose_stamp: float = 0.0

    # ------------------------------------------------------------------ 状态
    @property
    def landmarks(self) -> Dict[int, Landmark]:
        """当前地标（landmark_id -> Landmark）。"""
        return self._landmarks

    def reset(self) -> None:
        """清空全部观测与地标（SLAM 重定位 / 换场景时调用）。"""
        self._observations.clear()
        self._landmarks.clear()
        self._track_to_landmark.clear()
        self._next_landmark_id = 1
        self.last_pose = None
        self.last_pose_stamp = 0.0

    def set_last_pose(self, pose_wc: Optional[np.ndarray], stamp: float = 0.0) -> None:
        """记录最新相机位姿（用于没有位姿消息抵达时的兜底）。"""
        self.last_pose = None if pose_wc is None else np.asarray(pose_wc, dtype=np.float64).copy()
        self.last_pose_stamp = float(stamp)

    # ------------------------------------------------------------------ 主流程
    def add_detection(
        self,
        detection: Detection,
        stamp: float,
        pose_wc: Optional[np.ndarray] = None,
        frame_id: str = "",
        camera_matrix: Optional[np.ndarray] = None,
    ) -> Optional[Landmark]:
        """加入一次观测并尝试更新/新建地标。

        Args:
            detection: 检测结果（需带 ``track_id``）。
            stamp: 时间戳（秒）。
            pose_wc: 该帧相机位姿 ``(4, 4)`` 世界<-相机；``None`` 时用 :meth:`set_last_pose` 的值。
            frame_id: 位姿所在的参考系名（通常为 ``map``）。
            camera_matrix: 内参（用于由像素反算射线；若检测里已有 ``ray_camera`` 则不需要）。

        Returns:
            更新后的 :class:`Landmark`；观测不足或无法定位时返回 ``None``。
        """
        if detection.track_id is None:
            return None

        ray = self._ray_of(detection, camera_matrix)
        if ray is None:
            return None

        pose = pose_wc if pose_wc is not None else self.last_pose
        distance = None
        if detection.measurement is not None:
            distance = detection.measurement.distance_m

        observation = Observation(
            track_id=int(detection.track_id),
            stamp=float(stamp),
            ray_camera=ray,
            pose_wc=None if pose is None else np.asarray(pose, dtype=np.float64),
            distance_m=distance,
            confidence=float(detection.confidence),
            class_name=str(detection.class_name),
            center_px=detection.center,
            frame_id=frame_id,
        )

        history = self._observations.setdefault(observation.track_id, [])
        history.append(observation)
        if len(history) > int(self.config.max_observations):
            del history[: len(history) - int(self.config.max_observations)]

        return self._update_landmark(observation.track_id, history, camera_matrix)

    def prune(self, now: Optional[float] = None) -> List[int]:
        """删除超时未更新的地标（返回被删掉的 landmark_id 列表）。"""
        now = time.time() if now is None else float(now)
        removed: List[int] = []
        for landmark_id, landmark in list(self._landmarks.items()):
            if self.config.stale_after_s > 0 and (now - landmark.last_stamp) > self.config.stale_after_s:
                removed.append(landmark_id)
                del self._landmarks[landmark_id]
                track_id = landmark.track_id
                self._track_to_landmark.pop(track_id, None)
                self._observations.pop(track_id, None)
        return removed

    def snapshot(self) -> List[Landmark]:
        """返回当前所有地标（拷贝，供发布使用）。"""
        return list(self._landmarks.values())

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _ray_of(detection: Detection, camera_matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """取检测的相机系射线：优先用消息里带的，其次用内参反算。"""
        if detection.measurement is not None and detection.measurement.ray_camera is not None:
            ray = np.asarray(detection.measurement.ray_camera, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(ray))
            if norm > 1e-12:
                return ray / norm
        if camera_matrix is None:
            return None
        u, v = detection.center
        return pixel_to_ray(u, v, camera_matrix)

    def _update_landmark(
        self,
        track_id: int,
        history: Sequence[Observation],
        camera_matrix: Optional[np.ndarray],
    ) -> Optional[Landmark]:
        """尝试用历史观测三角化，并维护地标记录。"""
        posed = [obs for obs in history if obs.pose_wc is not None]

        if len(posed) >= max(2, int(self.config.min_views)):
            landmark = self._triangulate(track_id, posed, camera_matrix)
            if landmark is not None:
                self._landmarks[landmark.landmark_id] = landmark
                self._track_to_landmark[track_id] = landmark.landmark_id
                return landmark

        if self.config.allow_scale_fallback:
            fallback = self._scale_fallback(track_id, history)
            if fallback is not None:
                self._landmarks[fallback.landmark_id] = fallback
                self._track_to_landmark[track_id] = fallback.landmark_id
                return fallback
        return None

    def _triangulate(
        self,
        track_id: int,
        observations: Sequence[Observation],
        camera_matrix: Optional[np.ndarray],
    ) -> Optional[Landmark]:
        """多视图三角化 + 视差/基线门限 + 重投影误差剔除。"""
        cfg = self.config
        selected = self._filter_by_parallax(observations)
        if len(selected) < max(2, int(cfg.min_views)):
            return None

        origins = [obs.origin_world for obs in selected]
        directions = [obs.direction_world for obs in selected]
        if any(o is None for o in origins) or any(d is None for d in directions):
            return None

        point, metric_error, n_views = triangulate_rays(origins, directions)  # type: ignore[arg-type]
        if point is None:
            return None

        # 像素级重投影误差：需要内参 + 位姿，二者齐全时才能算
        px_error: Optional[float] = None
        used = list(selected)
        if camera_matrix is not None:
            errors: List[Tuple[Observation, float]] = []
            for obs in selected:
                error = self._reprojection_error_px(point, obs, camera_matrix)
                if error is not None:
                    errors.append((obs, error))
            if errors:
                px_error = float(math.sqrt(sum(e * e for _, e in errors) / len(errors)))
                inliers = [obs for obs, error in errors if error <= float(cfg.max_reproj_error_px)]
                if len(inliers) >= max(2, int(cfg.min_views)) and len(inliers) != len(selected):
                    origins = [obs.origin_world for obs in inliers]
                    directions = [obs.direction_world for obs in inliers]
                    refined, refined_error, refined_views = triangulate_rays(origins, directions)  # type: ignore[arg-type]
                    if refined is not None:
                        point, metric_error, n_views = refined, refined_error, refined_views
                        used = inliers
                        px_errors = [self._reprojection_error_px(point, obs, camera_matrix) for obs in inliers]
                        px_errors_valid = [e for e in px_errors if e is not None]
                        if px_errors_valid:
                            px_error = float(
                                math.sqrt(sum(e * e for e in px_errors_valid) / len(px_errors_valid))
                            )

        landmark_id = self._track_to_landmark.get(track_id)
        if landmark_id is None:
            landmark_id = self._next_landmark_id
            self._next_landmark_id += 1

        previous = self._landmarks.get(landmark_id)
        confidence = float(np.mean([obs.confidence for obs in used])) if used else 0.0
        distances = [obs.distance_m for obs in used if obs.distance_m is not None]

        return Landmark(
            landmark_id=landmark_id,
            track_id=track_id,
            position=np.asarray(point, dtype=np.float64).reshape(3),
            frame_id=used[0].frame_id or (previous.frame_id if previous else "map"),
            class_name=used[0].class_name,
            n_views=int(n_views),
            reproj_error_px=px_error,
            triangulation_error=float(metric_error),
            source="triangulated",
            confidence=confidence,
            first_stamp=previous.first_stamp if previous else min(obs.stamp for obs in used),
            last_stamp=max(obs.stamp for obs in used),
            distance_m=float(np.mean(distances)) if distances else None,
        )

    @staticmethod
    def _reprojection_error_px(point_world: np.ndarray, observation: Observation, camera_matrix: np.ndarray) -> Optional[float]:
        """把世界点投影回该观测的像素坐标，返回偏差（像素）。"""
        if observation.pose_wc is None:
            return None
        t_cw = invert_pose(observation.pose_wc)
        point_camera = t_cw[:3, :3] @ np.asarray(point_world).reshape(3) + t_cw[:3, 3]
        if point_camera[2] <= 1e-6:
            return None
        uvw = np.asarray(camera_matrix, dtype=np.float64) @ point_camera
        projected = uvw[:2] / uvw[2]
        return float(np.linalg.norm(projected - np.asarray(observation.center_px, dtype=np.float64)))

    def _filter_by_parallax(self, observations: Sequence[Observation]) -> List[Observation]:
        """按"与第一条观测的视差/基线"筛掉几乎同视角的观测（三角化病态）。"""
        cfg = self.config
        kept: List[Observation] = []
        for obs in observations:
            if not kept:
                kept.append(obs)
                continue
            base = kept[0]
            origin_a, origin_b = base.origin_world, obs.origin_world
            direction_a, direction_b = base.direction_world, obs.direction_world
            if origin_a is None or origin_b is None or direction_a is None or direction_b is None:
                continue
            baseline = float(np.linalg.norm(origin_b - origin_a))
            cos_angle = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
            parallax_deg = math.degrees(math.acos(cos_angle))
            if baseline < float(cfg.min_baseline_m):
                continue
            if parallax_deg < float(cfg.min_parallax_deg):
                continue
            kept.append(obs)
        return kept

    def _scale_fallback(self, track_id: int, history: Sequence[Observation]) -> Optional[Landmark]:
        """没有 SLAM 位姿时，用已知边长给出相机系下的近似位置。"""
        latest = history[-1]
        if latest.distance_m is None or latest.distance_m <= 0.0:
            return None
        position = latest.ray_camera.reshape(3) * float(latest.distance_m)
        landmark_id = self._track_to_landmark.get(track_id)
        if landmark_id is None:
            landmark_id = self._next_landmark_id
            self._next_landmark_id += 1
        previous = self._landmarks.get(landmark_id)
        return Landmark(
            landmark_id=landmark_id,
            track_id=track_id,
            position=position,
            frame_id=latest.frame_id or (previous.frame_id if previous else "camera"),
            class_name=latest.class_name,
            n_views=1,
            reproj_error_px=None,
            triangulation_error=None,
            source="scale_fallback",
            confidence=float(latest.confidence),
            first_stamp=previous.first_stamp if previous else latest.stamp,
            last_stamp=latest.stamp,
            distance_m=float(latest.distance_m),
        )
