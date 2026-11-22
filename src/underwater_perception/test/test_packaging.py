"""打包与集成一致性检查（纯静态，无需 ROS）。

为什么必须有这份测试：本包最终要在 **Ubuntu + ROS2** 上通过 ``colcon build`` 安装并被
``ros2 launch`` 启动，而开发机没有 ROS。以下三类错误都属于"本地看不出来、上机必炸"：

1. ``setup.py`` 的 ``console_scripts`` 名字与 launch 里 ``executable=`` 不一致
   -> ``ros2 launch`` 报 "executable 'xxx' not found"；
2. ``data_files`` 的 glob 没覆盖到实际目录（launch/config/docs）
   -> 安装后 ``share/<pkg>/launch`` 为空，同样启动失败；
3. ``resource/<pkg>`` 标记文件缺失 -> ``ament_index`` 找不到包。

这些都可以在这里静态查出来，因此写成测试。
"""

from __future__ import annotations

import ast
import configparser
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = PACKAGE_ROOT / "launch"
CONFIG_DIR = PACKAGE_ROOT / "config"
DOCS_DIR = PACKAGE_ROOT / "docs"


def _module_string_constants(tree: ast.Module) -> dict:
    """收集模块级字符串常量（例如 ``package_name = "underwater_perception"``）。"""
    constants: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


def _eval_string(node: ast.AST, constants: dict) -> str:
    """把一个表达式尽量还原成字符串：常量 / 变量 / 字符串拼接 / os.path.join / glob。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, "")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_string(node.left, constants) + _eval_string(node.right, constants)
    if isinstance(node, ast.Call):
        func = node.func
        name = getattr(func, "attr", "") or getattr(func, "id", "")
        parts = [_eval_string(argument, constants) for argument in node.args]
        if name == "join":
            return "/".join(part.strip("/") for part in parts if part)
        if name == "glob":
            return parts[0] if parts else ""
    return ""


def _setup_py_params() -> dict:
    """用 ast 解析 ``setup.py``，取出 ``name``、``data_files`` 与入口名（不执行它）。

    注意：``setup.py`` 里的值大多是**变量与函数调用**（``name=package_name``、
    ``os.path.join("share", package_name, "launch")``、``glob("launch/*.launch.py")``），
    因此这里必须做一个小型求值器。早先只取 ``ast.Constant``，结果解析出来是空的，
    测试自身失效（被下一个测试的断言暴露）。
    """
    source = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = _module_string_constants(tree)
    params: dict = {}

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "name":
                params["name"] = _eval_string(keyword.value, constants)
            elif keyword.arg == "data_files" and isinstance(keyword.value, ast.List):
                entries = []
                for element in keyword.value.elts:
                    if isinstance(element, ast.Tuple) and len(element.elts) == 2:
                        destination = _eval_string(element.elts[0], constants)
                        pattern = _eval_string(element.elts[1], constants)
                        entries.append((destination, pattern))
                params["data_files"] = entries
            elif keyword.arg == "entry_points" and isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    if not (isinstance(key, ast.Constant) and key.value == "console_scripts"):
                        continue
                    if not isinstance(value, ast.List):
                        continue
                    scripts = []
                    for item in value.elts:
                        text = _eval_string(item, constants)
                        if "=" in text:
                            scripts.append(text.split("=")[0].strip())
                    params["console_scripts"] = scripts
    return params


def _launch_executables() -> dict:
    """收集 launch 文件里 ``Node(package=..., executable=...)`` 的组合。"""
    found: dict = {}
    for launch_file in sorted(LAUNCH_DIR.glob("*.launch.py")):
        tree = ast.parse(launch_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name != "Node":
                continue
            package = executable = None
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    if keyword.arg == "package":
                        package = keyword.value.value
                    elif keyword.arg == "executable":
                        executable = keyword.value.value
            if package and executable:
                found.setdefault(package, set()).add(executable)
    return found


def test_setup_py_declares_expected_package_and_entry_points() -> None:
    """包名与五个节点的入口名必须齐全（launch 依赖它们）。"""
    params = _setup_py_params()
    assert params.get("name") == "underwater_perception"
    scripts = set(params.get("console_scripts", []))
    for expected in (
        "camera",
        "slam_viewer",
        "dataset_capture",
        "cube_detector",
        "calibration_node",
        "slam_input_bridge",
        "semantic_landmark",
        "image_sequence_player",
        "cube_detect_cli",
        "calibrate_cli",
        "calibrate_gui",
        "capture_dataset",
        "yolo_dataset_check",
        "perception_selftest",
    ):
        assert expected in scripts, f"setup.py 缺少入口: {expected}"


def test_launch_executables_exist_in_this_package() -> None:
    """launch 里引用的本包可执行文件必须都在 setup.py 的入口表里。"""
    declared = set(_setup_py_params().get("console_scripts", []))
    referenced = _launch_executables()
    ours = referenced.get("underwater_perception", set())
    assert ours, "launch 里没有引用本包的任何节点（检查 launch 文件）"
    missing = ours - declared
    assert not missing, f"launch 引用了 setup.py 未声明的可执行文件: {sorted(missing)}"


def test_launch_references_known_packages() -> None:
    """launch 里引用的外部包必须是我们已知存在的（避免拼错包名）。"""
    # 这些包名来自工作区实际存在的包或 ROS2 标准包
    known = {
        "underwater_perception",   # 本包
        "image_enhancer_cpp",      # 本项目既有包（可执行名 enhancer，已验证）
        "yolo_vrx_bridge",         # 本项目既有包（可执行名 yolo_detector，已验证）
        "ros2_orb_slam3",          # 第三方 ORB-SLAM3 封装（可执行名 mono_node_cpp，已验证）
        "v4l2_camera",             # ros-humble-v4l2-camera
        "rviz2",                   # ros-humble-rviz2
    }
    unknown = set(_launch_executables()) - known
    assert not unknown, f"launch 引用了未知的包名（可能拼错）: {sorted(unknown)}"


def test_data_files_cover_real_directories() -> None:
    """data_files 的 glob 必须真的能匹配到文件，否则安装后 share 目录是空的。"""
    params = _setup_py_params()
    data_files = params.get("data_files", [])
    assert data_files, "setup.py 没有声明 data_files（解析失败或真的缺失）"

    patterns = {pattern for _destination, pattern in data_files}
    for pattern, directory in (
        ("launch/*.launch.py", LAUNCH_DIR),
        ("config/*.yaml", CONFIG_DIR),
        ("config/orbslam3/*.yaml", CONFIG_DIR / "orbslam3"),
        ("config/rviz/*.rviz", CONFIG_DIR / "rviz"),
        ("docs/*.md", DOCS_DIR),
    ):
        assert pattern in patterns, f"data_files 里缺少 {pattern} 的安装项，实际有: {sorted(patterns)}"
        assert list(directory.glob(pattern.split("/")[-1])), f"{directory} 下没有匹配 {pattern} 的文件"

    # 目标路径必须落在 share/<包名>/ 下，否则 ros2 pkg prefix 找不到这些资源
    for destination, _pattern in data_files:
        assert destination == "share/ament_index/resource_index/packages" or destination.startswith(
            "share/underwater_perception"
        ), f"data_files 安装目标异常: {destination}"


def test_ament_index_resource_marker_exists() -> None:
    """``resource/<包名>`` 标记文件必须存在（ament_index 靠它发现包）。"""
    marker = PACKAGE_ROOT / "resource" / "underwater_perception"
    assert marker.is_file(), "缺少 resource/underwater_perception 标记文件"


def test_setup_cfg_installs_scripts_into_package_lib() -> None:
    """``setup.cfg`` 必须把脚本装到 ``lib/<包名>``，否则 ``ros2 run`` 找不到节点。"""
    parser = configparser.ConfigParser()
    parser.read(PACKAGE_ROOT / "setup.cfg", encoding="utf-8")
    assert parser.has_section("install")
    assert parser.get("install", "install_scripts") == "$base/lib/underwater_perception"


def test_package_xml_declares_all_runtime_dependencies() -> None:
    """节点用到的消息包都必须在 ``package.xml`` 里声明（rosdep/构建依赖）。"""
    text = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
    for dependency in (
        "rclpy",
        "std_msgs",
        "std_srvs",
        "sensor_msgs",
        "geometry_msgs",
        "nav_msgs",
        "visualization_msgs",
        "python3-numpy",
        "python3-opencv",
        "python3-yaml",
    ):
        assert f"<exec_depend>{dependency}</exec_depend>" in text, f"package.xml 缺少依赖: {dependency}"


def test_ros_node_sources_do_not_import_rclpy_from_core() -> None:
    """分层红线：``core/`` 里不允许出现 rclpy（否则离线测试能力就没了）。"""
    core_dir = PACKAGE_ROOT / "underwater_perception" / "core"
    offenders = []
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [path.name for alias in node.names if alias.name.split(".")[0] == "rclpy"]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "rclpy":
                offenders.append(path.name)
    assert not offenders, f"core/ 不允许 import rclpy: {sorted(set(offenders))}"


@pytest.mark.parametrize("module", ["nodes", "tools"])
def test_non_core_modules_compile(module: str) -> None:
    """节点与工具目录的模块必须语法合法（无法 import，但可以编译）。"""
    directory = PACKAGE_ROOT / "underwater_perception" / module
    files = sorted(directory.glob("*.py"))
    assert files, f"{module}/ 下没有模块"
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
