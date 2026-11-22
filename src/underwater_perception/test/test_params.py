"""参数系统单元测试：round-trip、YAML 解包、预设解析。

这些测试的目的很实际：现场最常见的故障就是"我在 yaml 里改了参数但没生效"，
根因永远是"写进去的键名"和"读出来的键名"不一致。``test_params_roundtrip``
就是专门盯这个问题的回归测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from underwater_perception.core.params import (
    HSV_PRESETS,
    CalibrationConfig,
    ColorConfig,
    CubeDetectorConfig,
    LandmarkConfig,
    SlamBridgeConfig,
    dump_flat_yaml,
    load_flat_yaml,
)


def test_params_roundtrip_is_lossless() -> None:
    """from_flat -> to_flat -> from_flat 必须完全一致（键名对齐的回归测试）。"""
    flat = CubeDetectorConfig().to_flat()
    once = CubeDetectorConfig.from_flat(flat)
    twice = CubeDetectorConfig.from_flat(once.to_flat())

    first, second = once.to_flat(), twice.to_flat()
    assert set(first) == set(second), f"键集合不一致: {set(first) ^ set(second)}"
    for key in first:
        assert first[key] == pytest.approx(second[key]) if isinstance(first[key], float) else first[key] == second[key], (
            f"参数 {key} 在 round-trip 后发生变化: {first[key]} != {second[key]}"
        )


def test_hsv_ranges_override_preset() -> None:
    """显式 hsv_ranges 必须覆盖 preset（现场微调时必须生效）。"""
    config = ColorConfig(preset="blue", hsv_ranges=[0, 100, 50, 8, 255, 255])
    ranges = config.resolved_ranges()
    assert len(ranges) == 1
    assert ranges[0] == (0, 100, 50, 8, 255, 255)
    assert config.resolved_ranges_flat() == [0, 100, 50, 8, 255, 255]


def test_red_preset_has_two_ranges() -> None:
    """红色的色相跨越 0 度，必须是两段区间，否则会漏检。"""
    ranges = ColorConfig(preset="red").resolved_ranges()
    assert len(ranges) == 2
    assert ranges[0][0] == 0
    assert ranges[1][3] >= 170
    assert "red" in HSV_PRESETS


def test_load_flat_yaml_accepts_ros_and_flat_formats(tmp_path: Path) -> None:
    """同一个文件既能被 ROS 读（ros__parameters），也能被 CLI 读（自动解包）。"""
    ros_style = tmp_path / "ros.yaml"
    ros_style.write_text(
        "cube_detector:\n  ros__parameters:\n    min_area_px: 321\n    mode: cube_silhouette\n",
        encoding="utf-8",
    )
    flat = load_flat_yaml(ros_style)
    assert flat["min_area_px"] == 321
    assert flat["mode"] == "cube_silhouette"

    explicit = load_flat_yaml(ros_style, node_name="cube_detector")
    assert explicit["min_area_px"] == 321

    plain = tmp_path / "plain.yaml"
    plain.write_text("min_area_px: 55\ncolor_preset: red\n", encoding="utf-8")
    assert load_flat_yaml(plain)["min_area_px"] == 55

    missing = tmp_path / "nope.yaml"
    assert load_flat_yaml(missing) == {}


def test_dump_flat_yaml_is_reloadable(tmp_path: Path) -> None:
    """写出的 ROS 参数文件必须能被自己读回来（用于 --dump-config 与报告）。"""
    flat = CubeDetectorConfig().to_flat()
    text = dump_flat_yaml(flat, "cube_detector")
    path = tmp_path / "dumped.yaml"
    path.write_text(text, encoding="utf-8")
    reloaded = load_flat_yaml(path)
    assert reloaded["min_area_px"] == flat["min_area_px"]
    assert reloaded["hsv_ranges"] == flat["hsv_ranges"]


def test_measure_config_requires_full_intrinsics() -> None:
    """内参不全时必须返回 None，而不是拿半套参数算出一个错误距离。"""
    assert CubeDetectorConfig().measure.camera_matrix() is None
    config = CubeDetectorConfig.from_flat(
        {"camera_fx": 700.0, "camera_fy": 700.0, "camera_cx": 640.0, "camera_cy": 240.0}
    )
    matrix = config.measure.camera_matrix()
    assert matrix is not None
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(700.0)
    assert config.measure.distortion().shape == (1, 5)


def test_other_configs_have_sane_defaults() -> None:
    """其余配置的默认值必须符合实物情况（8x6 内角点 / 25 mm / 红色立方体）。"""
    calib = CalibrationConfig()
    assert calib.pattern_size == (8, 6)
    assert calib.square_size == pytest.approx(0.025)

    detector = CubeDetectorConfig()
    assert detector.color.preset == "red"
    assert detector.class_name == "cube"
    assert detector.measure.cube_edge_m == pytest.approx(0.05)

    bridge = SlamBridgeConfig.from_flat({})
    assert bridge.target_image_topic == "/mono_py_driver/img_msg"
    assert bridge.settings_name == "water_1280x480"

    landmark = LandmarkConfig.from_flat({})
    assert landmark.min_views == 2
    assert landmark.min_parallax_deg > 0.0


def test_landmark_class_whitelist_and_confidence_filter() -> None:
    """真实 YOLO 多类别时，白名单必须拦住动态目标。"""
    config = LandmarkConfig.from_flat(
        {"landmark_class_names": "cube, marker", "min_detection_confidence": 0.35}
    )
    assert config.landmark_class_names == ["cube", "marker"]
    assert config.accepts("cube", 0.9) is True
    assert config.accepts("marker", 0.35) is True
    assert config.accepts("fish", 0.99) is False
    assert config.accepts("cube", 0.34) is False

    # 空列表 = 不过滤
    assert LandmarkConfig.from_flat({}).accepts("fish", 0.1) is True
