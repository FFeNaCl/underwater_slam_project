"""检测消息契约与几何工具的单元测试（纯 Python，不需要 ROS）。"""

from __future__ import annotations

import json

import pytest

from underwater_perception.core.types import (
    SCHEMA_VERSION,
    Detection,
    DetectionFrame,
    GeometryMetrics,
    Measurement,
    bbox_iou,
    polygon_area,
    polygon_iou,
    xyxy_from_polygon,
)


def test_detection_roundtrip_preserves_core_fields() -> None:
    """检测对象序列化/反序列化后核心字段必须完全一致。"""
    detection = Detection(
        class_id=0,
        class_name="cube",
        confidence=0.83,
        xyxy=[10.0, 20.0, 60.0, 70.0],
        score_terms={"fill": 0.9},
        track_id=3,
        geometry=GeometryMetrics(center=(35.0, 45.0), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], vertex_count=4),
        measurement=Measurement(distance_m=0.5, bearing_rad=(0.1, 0.2), ray_camera=[0.1, 0.2, 0.97], method="known_size"),
    )
    restored = Detection.from_dict(detection.to_dict())
    assert restored.class_id == detection.class_id
    assert restored.class_name == detection.class_name
    assert restored.confidence == pytest.approx(detection.confidence)
    assert restored.xyxy == detection.xyxy
    assert restored.track_id == 3
    assert restored.geometry is not None and restored.geometry.vertex_count == 4
    assert restored.measurement is not None and restored.measurement.distance_m == pytest.approx(0.5)


def test_frame_json_is_superset_of_legacy_yolo_schema() -> None:
    """帧 JSON 必须包含老 YOLO 节点的全部字段，保证下游可无缝替换。"""
    frame = DetectionFrame(
        detections=[Detection(class_id=1, class_name="cube", confidence=0.5, xyxy=[1, 2, 3, 4])],
        stamp_sec=12,
        stamp_nanosec=34,
        frame_id="camera",
        image_width=640,
        image_height=480,
        source="geometry",
    )
    payload = json.loads(frame.to_json())
    for key in ("stamp", "frame_id", "source", "count", "detections"):
        assert key in payload, f"缺少老节点也有的字段: {key}"
    assert payload["stamp"] == {"sec": 12, "nanosec": 34}
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["image"] == {"width": 640, "height": 480}
    assert payload["detections"][0]["xyxy"] == [1.0, 2.0, 3.0, 4.0]


def test_frame_can_parse_legacy_yolo_json() -> None:
    """必须能解析现有 yolo_detector_node 产生的 JSON（没有 schema_version / image）。"""
    legacy = json.dumps(
        {
            "stamp": {"sec": 1, "nanosec": 2},
            "frame_id": "vrx_sim_camera",
            "source": "model",
            "count": 1,
            "detections": [
                {"class_id": 2, "class_name": "fishing_boat", "confidence": 0.7, "xyxy": [5, 6, 7, 8]}
            ],
        }
    )
    frame = DetectionFrame.from_json(legacy)
    assert frame.source == "model"
    assert frame.frame_id == "vrx_sim_camera"
    assert len(frame.detections) == 1
    assert frame.detections[0].class_name == "fishing_boat"


def test_geometry_helpers() -> None:
    """面积 / 外接框 / IoU 的基本正确性。"""
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_area(square) == pytest.approx(100.0)
    assert xyxy_from_polygon(square) == [0.0, 0.0, 10.0, 10.0]
    assert polygon_iou(square, square) == pytest.approx(1.0, abs=0.02)
    assert polygon_iou(square, [[100, 100], [110, 100], [110, 110], [100, 110]]) == pytest.approx(0.0, abs=1e-6)
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25.0 / 175.0, rel=1e-6)
    assert bbox_iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_polygon_iou_handles_degenerate_input() -> None:
    """退化输入必须返回 0 而不是抛异常（节点里不能因为脏数据崩）。"""
    assert polygon_iou([[0, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]]) == 0.0
    assert polygon_iou([[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [1, 1]]) == 0.0
