"""静态一致性检查：配置文件键名、launch 参数声明、模块语法。

为什么需要这些测试：本项目在 Windows 上开发、在 Ubuntu + ROS2 上运行，
**没有 ROS 时无法启动节点**。而配置/launch 类错误恰恰是"运行才炸"的高发区：

* 配置文件里键名拼错 -> 参数默默不生效（最难查的一类问题，现场表现为"我明明改了没用"）
* launch 里用了没有 ``DeclareLaunchArgument`` 声明的 ``LaunchConfiguration``
  -> launch 执行到该 Node 时报错（开发时就真实踩过：``use_real_camera`` 未声明）
* 节点/工具里的语法错误 -> 只有 import 时才暴露，而 import 需要 rclpy

这三类都可以用纯文本/ast 静态检查出来，因此放在这里。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from underwater_perception.core.params import CubeDetectorConfig, load_flat_yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"
LAUNCH_DIR = PACKAGE_ROOT / "launch"
NODES_DIR = PACKAGE_ROOT / "underwater_perception" / "nodes"
TOOLS_DIR = PACKAGE_ROOT / "underwater_perception" / "tools"


def _node_default_keys(node_file: str) -> set:
    """从节点源码里提取 ``defaults = {...}`` 的字面量键名（不 import rclpy）。"""
    source = (NODES_DIR / node_file).read_text(encoding="utf-8")
    tree = ast.parse(source)
    keys: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "defaults" not in targets and "flat_defaults" not in targets:
                continue
            if isinstance(node.value, ast.Dict):
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        # flat_defaults.update({...}) 形式
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    for key in arg.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            keys.add(key.value)
    return keys


def test_cube_detector_config_keys_are_all_recognised() -> None:
    """cube_detector.yaml 里的每个键都必须能被参数系统识别（否则等于没配）。"""
    yaml_path = CONFIG_DIR / "cube_detector.yaml"
    assert yaml_path.is_file(), "缺少 config/cube_detector.yaml"

    values = load_flat_yaml(yaml_path, node_name="cube_detector")
    assert values, "配置文件解析为空"
    known = set(CubeDetectorConfig().to_flat().keys())
    unknown = set(values) - known
    assert not unknown, f"cube_detector.yaml 里有无法识别的键（会静默失效）: {sorted(unknown)}"

    # 反向检查：核心参数必须在文件里显式出现，避免"文件里删掉某个键导致行为变化"
    for key in ("color_preset", "hsv_ranges", "mode", "min_confidence", "cube_edge_m", "use_silhouette_fit"):
        assert key in values, f"cube_detector.yaml 缺少关键参数: {key}"

    # 文件里的值必须能构造出合法配置
    config = CubeDetectorConfig.from_flat(values)
    assert config.color.preset == "red"
    assert len(config.color.resolved_ranges()) == 2, "红色必须解析出两段 HSV 区间"
    assert config.measure.cube_edge_m > 0.0


@pytest.mark.parametrize(
    "yaml_name,node_file,node_name",
    [
        ("calibration.yaml", "calibration_node.py", "calibration_node"),
        ("slam_bridge.yaml", "slam_input_bridge_node.py", "slam_input_bridge"),
        ("semantic_landmark.yaml", "semantic_landmark_node.py", "semantic_landmark"),
        ("dataset_capture.yaml", "dataset_capture_node.py", "dataset_capture"),
    ],
)
def test_node_config_keys_match_declared_defaults(yaml_name: str, node_file: str, node_name: str) -> None:
    """每个节点配置文件里的键，必须都是该节点真正声明过的参数。"""
    yaml_path = CONFIG_DIR / yaml_name
    assert yaml_path.is_file(), f"缺少 config/{yaml_name}"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert node_name in raw, f"{yaml_name} 里缺少节点块 {node_name}"
    params = raw[node_name].get("ros__parameters", {})
    assert params, f"{yaml_name} 的 ros__parameters 为空"

    declared = _node_default_keys(node_file)
    assert declared, f"未能从 {node_file} 解析出参数默认值（检查 defaults/flat_defaults 的写法）"
    unknown = set(params) - declared
    assert not unknown, f"{yaml_name} 里有 {node_file} 未声明的参数: {sorted(unknown)}"


def _launch_python_files() -> list:
    return sorted(LAUNCH_DIR.glob("*.launch.py"))


@pytest.mark.parametrize("launch_file", _launch_python_files(), ids=lambda p: p.name)
def test_launch_files_parse_and_declare_their_arguments(launch_file: Path) -> None:
    """launch 文件必须语法正确，且用到的 LaunchConfiguration 都已声明。

    规则来源：ROS2 launch 在执行 Node 时会去取 ``LaunchConfiguration`` 的值，
    若该名字没有对应的 ``DeclareLaunchArgument``（也没有外部传入），
    执行到这一步就会失败。开发过程中真实踩过这个坑。
    """
    source = launch_file.read_text(encoding="utf-8")
    tree = ast.parse(source)  # 语法错误会在这里直接抛 SyntaxError

    used: set = set()
    declared: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
            if name == "LaunchConfiguration" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    used.add(first.value)
            if name == "DeclareLaunchArgument" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    declared.add(first.value)

    assert declared, f"{launch_file.name} 没有声明任何 launch 参数"
    undeclared = used - declared
    assert not undeclared, f"{launch_file.name} 使用了未声明的 launch 参数: {sorted(undeclared)}"


def test_launch_files_reference_existing_config_files() -> None:
    """launch 里通过 ``os.path.join(config_dir, ...)`` 引用的配置文件必须真实存在。

    正则要匹配**多段拼接**（例如 ``os.path.join(config_dir, "rviz", "slam.rviz")``），
    早先只取第一段，会把目录 "rviz" 当成文件判定失败（测试自身的 bug）。
    """
    pattern = re.compile(r"os\.path\.join\(\s*config_dir\s*,([^)]*)\)")
    for launch_file in _launch_python_files():
        source = launch_file.read_text(encoding="utf-8")
        for argument_list in pattern.findall(source):
            parts = [part.strip().strip('"').strip("'") for part in argument_list.split(",")]
            parts = [part for part in parts if part]
            if not parts:
                continue
            target = CONFIG_DIR.joinpath(*parts)
            assert target.exists(), f"{launch_file.name} 引用了不存在的配置: {target.relative_to(CONFIG_DIR.parent)}"


def test_all_python_modules_compile() -> None:
    """所有模块都必须能通过语法编译（Windows 上没有 rclpy，无法真正 import）。"""
    targets = list(NODES_DIR.glob("*.py")) + list(TOOLS_DIR.glob("*.py"))
    targets += list((PACKAGE_ROOT / "underwater_perception" / "core").glob("*.py"))
    assert targets, "没有找到待检查的模块"
    for path in targets:
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - 出错即测试失败
            pytest.fail(f"{path.name} 语法错误: {exc}")


def test_orbslam3_settings_template_has_required_keys() -> None:
    """ORB-SLAM3 设置模板必须含全部必需键（键名区分大小写，缺一个就启动失败）。"""
    template = CONFIG_DIR / "orbslam3" / "water_1280x480.yaml"
    assert template.is_file()
    text = template.read_text(encoding="utf-8")
    for key in (
        "Camera.type",
        "Camera.fx",
        "Camera.fy",
        "Camera.cx",
        "Camera.cy",
        "Camera.k1",
        "Camera.k2",
        "Camera.p1",
        "Camera.p2",
        "Camera.k3",
        "Camera.width",
        "Camera.height",
        "Camera.fps",
        "Camera.RGB",
        "ORBextractor.nFeatures",
        "ORBextractor.scaleFactor",
        "ORBextractor.nLevels",
        "ORBextractor.iniThFAST",
        "ORBextractor.minThFAST",
    ):
        assert f"{key}:" in text, f"ORB-SLAM3 设置模板缺少键: {key}"
    assert text.lstrip().startswith("%YAML:1.0"), "ORB-SLAM3 设置文件必须以 %YAML:1.0 开头"
    assert 'Camera.type: "PinHole"' in text


def test_rviz_configs_have_matching_fixed_frames() -> None:
    """RViz 配置的 Fixed Frame 必须与节点发布的 frame_id 约定一致。"""
    detection = (CONFIG_DIR / "rviz" / "detection.rviz").read_text(encoding="utf-8")
    slam = (CONFIG_DIR / "rviz" / "slam.rviz").read_text(encoding="utf-8")
    assert "Fixed Frame: camera_optical" in detection
    assert "Fixed Frame: map" in slam
    # SLAM 配置必须包含轨迹与点云显示，否则"看不到建图结果"会被误判成 SLAM 没工作
    assert "/slam/path" in slam
    assert "/slam/map_points" in slam
    assert "/semantic/markers" in slam
