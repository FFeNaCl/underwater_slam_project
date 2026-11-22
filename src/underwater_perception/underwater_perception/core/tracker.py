"""跨帧 IoU 跟踪。

为什么几何旁路也需要跟踪：
1. 单帧误检（反光斑、红色杂物）靠 ``min_hits`` 可以压掉；
2. 语义地标需要**稳定 ID**才能把多帧观测关联到同一个物理立方体；
3. 演示时画面上 ID 稳定，观感与可解释性都更好。

跟踪器是纯 Python 状态机，不依赖 ROS，方便离线用合成序列测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .params import TrackConfig
from .types import Detection, bbox_iou, polygon_iou

__all__ = ["Track", "IoUTracker"]


@dataclass
class Track:
    """一条轨迹（一个假设的物理目标）。"""

    track_id: int
    class_id: int
    class_name: str
    xyxy: List[float]
    polygon: List[List[float]] = field(default_factory=list)
    confidence: float = 0.0
    hits: int = 0
    age: int = 0
    misses: int = 0
    last_detection: Optional[Detection] = None
    distance_m: Optional[float] = None
    #: 该轨迹累计被观测到的次数（含未确认帧），供地标节点估计观测数
    observations: int = 0
    #: 是否已确认（由跟踪器按 ``min_hits`` 置位）
    confirmed: bool = False

    def center(self) -> Tuple[float, float]:
        """轨迹外接框中心（像素）。"""
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class IoUTracker:
    """基于 IoU 的贪心匹配跟踪器（SORT 的简化版，无卡尔曼预测）。

    选择贪心 + IoU 而不是卡尔曼滤波的原因：水下立方体运动慢、帧率稳定，
    IoU 关联足够；而且纯几何、零超参，现场不需要调。
    若将来要跟踪快速运动的鱼群，再引入匀速模型。

    Args:
        config: 跟踪参数；``None`` 时用默认值（开启，min_hits=2）。
    """

    def __init__(self, config: Optional[TrackConfig] = None) -> None:
        self.config = config or TrackConfig()
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frame_index = 0

    @property
    def tracks(self) -> Dict[int, Track]:
        """当前所有轨迹（含未确认）。"""
        return self._tracks

    def reset(self) -> None:
        """清空跟踪状态（SLAM 重定位或换场景时调用）。"""
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0

    def update(self, detections: Sequence[Detection]) -> List[Detection]:
        """用当前帧检测结果更新轨迹，返回**带 track_id 的已确认检测**。

        注意：本方法会就地写入 ``detections[i].track_id``。

        Args:
            detections: 当前帧检测结果。

        Returns:
            已分配 ``track_id`` 的检测；``publish_unconfirmed=False`` 时只返回确认轨迹。
        """
        self._frame_index += 1
        cfg = self.config
        if not cfg.enabled:
            return list(detections)

        # 贪心匹配：把所有 (轨迹, 检测) 组合按 IoU 从高到低排序后依次占用
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_index, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                score = self._association_score(track, det)
                if score >= cfg.iou_threshold:
                    candidates.append((score, track_id, det_index))
        candidates.sort(reverse=True)

        used_tracks: set = set()
        used_dets: set = set()
        for score, track_id, det_index in candidates:
            if track_id in used_tracks or det_index in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_index)
            track = self._tracks[track_id]
            det = detections[det_index]
            track.xyxy = list(det.xyxy)
            track.polygon = det.polygon()
            track.confidence = det.confidence
            track.hits += 1
            track.observations += 1
            track.misses = 0
            track.age += 1
            track.last_detection = det
            if det.measurement is not None and det.measurement.distance_m is not None:
                track.distance_m = det.measurement.distance_m
            if track.hits >= cfg.min_hits:
                track.confirmed = True
            det.track_id = track.track_id

        # 未匹配的检测 -> 新建轨迹（尚未确认）
        for det_index, det in enumerate(detections):
            if det_index in used_dets:
                continue
            track = Track(
                track_id=self._next_id,
                class_id=det.class_id,
                class_name=det.class_name,
                xyxy=list(det.xyxy),
                polygon=det.polygon(),
                confidence=det.confidence,
                hits=1,
                age=1,
                observations=1,
                last_detection=det,
                confirmed=cfg.min_hits <= 1,
            )
            det.track_id = track.track_id
            self._tracks[track.track_id] = track
            self._next_id += 1

        # 未匹配的轨迹 -> 计一次丢失
        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            track.misses += 1
            track.age += 1

        # 淘汰过期轨迹
        for track_id in [tid for tid, t in self._tracks.items() if t.misses > cfg.max_age]:
            del self._tracks[track_id]

        # 组装输出
        out: List[Detection] = []
        for det in detections:
            if det.track_id is None:
                continue
            track = self._tracks.get(det.track_id)
            if track is None:
                continue
            if not track.confirmed and not cfg.publish_unconfirmed:
                continue
            out.append(det)
        return out

    def _association_score(self, track: Track, det: Detection) -> float:
        """轨迹与检测的关联分数：多边形 IoU 优先，退化为外接框 IoU。"""
        score = 0.0
        if len(track.polygon) >= 3:
            score = polygon_iou(track.polygon, det.polygon())
        if score <= 0.0:
            score = bbox_iou(track.xyxy, det.xyxy)

        limit = self.config.max_center_jump_px
        if limit > 0.0:
            tcx, tcy = track.center()
            dcx, dcy = det.center
            if math.hypot(dcx - tcx, dcy - tcy) > limit:
                return 0.0
        return float(score)
