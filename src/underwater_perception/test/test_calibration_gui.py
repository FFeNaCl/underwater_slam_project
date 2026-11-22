"""标定 GUI 的纯函数测试（不创建窗口，无显示器也能跑）。"""

from __future__ import annotations

import numpy as np

from underwater_perception.core.calibration import ViewResult
from underwater_perception.tools.calibrate_gui import (
    board_summary,
    build_calibration_config,
    is_novel_view,
    validate_board_params,
)


def _view(corners: np.ndarray) -> ViewResult:
    return ViewResult(
        corners=np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        object_points=None,
        image_size=(640, 480),
        accepted=True,
    )


def test_board_summary_chessboard_uses_inner_corners_and_squares() -> None:
    info = board_summary("chessboard", 8, 6, 25.0)
    assert info["inner_corners"] == "8x6"
    # 8x6 内角点 = 9x7 个方格
    assert info["squares"] == "9x7"
    assert info["physical_mm"] == (225.0, 175.0)
    assert info["object_points"] == 48


def test_board_summary_circles_uses_spacing_not_extra_square() -> None:
    info = board_summary("circles", 8, 6, 25.0)
    assert info["squares"] == "8x6"
    assert info["physical_mm"] == (175.0, 125.0)


def test_validate_board_params_rejects_impossible_values() -> None:
    errors, _warnings = validate_board_params("chessboard", 2, 6, 25.0)
    assert errors

    errors, _warnings = validate_board_params("chessboard", 8, 6, 0.0)
    assert errors

    errors, warnings = validate_board_params("chessboard", 8, 6, 3.0)
    assert not errors
    assert warnings  # 3 mm 太小，只警告不阻止

    # 把 25 mm 写成 0.025（米）是最常见的手误，必须拦下
    errors, _warnings = validate_board_params("chessboard", 8, 6, 0.025)
    assert errors
    assert "毫米" in errors[0]


def test_build_calibration_config_converts_mm_to_m() -> None:
    config = build_calibration_config(
        pattern="chessboard",
        cols=8,
        rows=6,
        square_mm=25.0,
        output_dir="/tmp/calib",
        camera_name="cam",
        orbslam3_name="water",
        fps=30.0,
        min_views=12,
        max_views=40,
    )
    assert config.square_size == 0.025
    assert config.pattern_size == (8, 6)
    assert config.output_dir == "/tmp/calib"


def test_is_novel_view_rejects_duplicate_and_accepts_shifted_view() -> None:
    grid = np.array([[x * 20.0, y * 20.0] for y in range(6) for x in range(8)], dtype=np.float32)
    first = _view(grid)
    duplicate = _view(grid)
    shifted = _view(grid + np.array([60.0, 0.0], dtype=np.float32))

    assert is_novel_view(first, []) is True
    assert is_novel_view(duplicate, [first]) is False
    assert is_novel_view(shifted, [first]) is True
