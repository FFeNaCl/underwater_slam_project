"""ament_python 打包脚本。

约定（非常重要，Linux 侧 AI 请勿改动结构）：
* `underwater_perception.core.*` 是纯算法层：只依赖 numpy / opencv / pyyaml，
  不 import rclpy，因此可以在没有 ROS 的机器上直接 pytest 与 CLI 运行。
* `underwater_perception.nodes.*` 是 ROS2 薄封装：只做参数声明、消息转换、
  调用 core，不放算法细节。
* `underwater_perception.tools.*` 是命令行工具与自测入口。
"""

import os

from glob import glob

from setuptools import find_packages, setup

package_name = "underwater_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "orbslam3"), glob("config/orbslam3/*.yaml")),
        (os.path.join("share", package_name, "config", "rviz"), glob("config/rviz/*.rviz")),
        (os.path.join("share", package_name, "docs"), glob("docs/*.md")),
        (os.path.join("share", package_name, "data", "euroc_sample"), glob("data/euroc_sample/*.png")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hu",
    maintainer_email="hu@example.com",
    description="水下感知：几何旁路立方体识别、相机标定、SLAM 桥接与语义地标",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ROS2 节点
            "camera = underwater_perception.nodes.camera_node:main",
            "slam_viewer = underwater_perception.nodes.slam_viewer_node:main",
            "cube_detector = underwater_perception.nodes.cube_detector_node:main",
            "calibration_node = underwater_perception.nodes.calibration_node:main",
            "slam_input_bridge = underwater_perception.nodes.slam_input_bridge_node:main",
            "semantic_landmark = underwater_perception.nodes.semantic_landmark_node:main",
            "image_sequence_player = underwater_perception.nodes.image_sequence_player_node:main",
            "dataset_capture = underwater_perception.nodes.dataset_capture_node:main",
            # 无 ROS 也能用的命令行工具
            "cube_detect_cli = underwater_perception.tools.cube_detect_cli:main",
            "calibrate_cli = underwater_perception.tools.calibrate_cli:main",
            "calibrate_gui = underwater_perception.tools.calibrate_gui:main",
            "capture_dataset = underwater_perception.tools.capture_dataset:main",
            "yolo_dataset_check = underwater_perception.tools.yolo_dataset_check:main",
            "perception_selftest = underwater_perception.tools.selftest:main",
        ],
    },
)
