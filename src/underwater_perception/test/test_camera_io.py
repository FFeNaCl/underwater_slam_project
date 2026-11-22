"""相机格式工具测试（不访问真实硬件）。"""

from __future__ import annotations

import cv2

from underwater_perception.core.camera_io import fourcc_to_str


def test_fourcc_to_str_roundtrip() -> None:
    for code in ("MJPG", "YUYV"):
        value = cv2.VideoWriter_fourcc(*code)
        assert fourcc_to_str(value) == code


def test_fourcc_to_str_handles_invalid_input() -> None:
    assert fourcc_to_str(None) == ""  # type: ignore[arg-type]
    assert fourcc_to_str("bad") == ""  # type: ignore[arg-type]
