"""IOU 跟踪器的单元测试。"""

from __future__ import annotations

from underwater_perception.core.params import TrackConfig
from underwater_perception.core.tracker import IoUTracker
from underwater_perception.core.types import Detection, GeometryMetrics


def _detection(x: float, y: float, size: float = 40.0, class_id: int = 0, confidence: float = 0.8) -> Detection:
    return Detection(
        class_id=class_id,
        class_name="cube",
        confidence=confidence,
        xyxy=[x - size / 2, y - size / 2, x + size / 2, y + size / 2],
        geometry=GeometryMetrics(
            center=(x, y),
            corners=[
                [x - size / 2, y - size / 2],
                [x + size / 2, y - size / 2],
                [x + size / 2, y + size / 2],
                [x - size / 2, y + size / 2],
            ],
            vertex_count=4,
        ),
    )


def test_requires_min_hits_before_publishing() -> None:
    """min_hits=3 时必须前两帧不发布、第三帧才发布。"""
    tracker = IoUTracker(TrackConfig(min_hits=3))
    assert tracker.update([_detection(100, 100)]) == []
    assert tracker.update([_detection(105, 100)]) == []
    published = tracker.update([_detection(110, 100)])
    assert len(published) == 1
    assert published[0].track_id == 1


def test_track_id_survives_short_occlusion() -> None:
    """目标短暂消失（不超过 max_age）后回来应保持同一 ID。"""
    tracker = IoUTracker(TrackConfig(min_hits=1, max_age=3, iou_threshold=0.1))
    tracker.update([_detection(100, 100)])
    for _ in range(2):  # 两帧丢失
        assert tracker.update([]) == []
    published = tracker.update([_detection(102, 100)])
    assert len(published) == 1
    assert published[0].track_id == 1, "短暂遮挡后 ID 不应变化"


def test_track_is_dropped_after_max_age() -> None:
    """超过 max_age 未出现的目标必须被淘汰，回来时得到新 ID。"""
    tracker = IoUTracker(TrackConfig(min_hits=1, max_age=2, iou_threshold=0.1))
    tracker.update([_detection(100, 100)])
    for _ in range(4):
        tracker.update([])
    published = tracker.update([_detection(100, 100)])
    assert len(published) == 1
    assert published[0].track_id == 2, "过期轨迹应被淘汰并分配新 ID"


def test_two_targets_get_separate_ids() -> None:
    """两个距离较远的目标必须各自有独立 ID。"""
    tracker = IoUTracker(TrackConfig(min_hits=1, iou_threshold=0.1))
    published = tracker.update([_detection(100, 100), _detection(400, 300)])
    assert len(published) == 2
    assert len({d.track_id for d in published}) == 2


def test_class_mismatch_does_not_associate() -> None:
    """不同类别不能被关联到同一条轨迹（避免 YOLO/几何两路互相污染）。"""
    tracker = IoUTracker(TrackConfig(min_hits=1, iou_threshold=0.05))
    first = tracker.update([_detection(100, 100, class_id=0)])
    assert first[0].track_id == 1
    second = tracker.update([_detection(100, 100, class_id=1)])
    assert len(second) == 1
    assert second[0].track_id != 1


def test_disabled_tracker_passes_through() -> None:
    """关闭跟踪时检测应原样通过（只是没有 track_id）。"""
    tracker = IoUTracker(TrackConfig(enabled=False))
    published = tracker.update([_detection(100, 100)])
    assert len(published) == 1
    assert published[0].track_id is None
