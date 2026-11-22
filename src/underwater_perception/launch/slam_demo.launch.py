"""无相机验证：用一段有运动的花卷/图片序列驱动完整 SLAM 链路。

默认使用 ORB-SLAM3 自带的 EuRoC MH05 测试序列（585 张 752x480），
用来验证「图像 -> 增强 -> 桥接 -> ORB-SLAM3 -> 轨迹/点云 -> RViz」是否真的通。
真实水下实验请用 ``mapping.launch.py``。

用法::

    ros2 launch underwater_perception slam_demo.launch.py

    # 换成自己的图片目录（按文件名排序）
    ros2 launch underwater_perception slam_demo.launch.py \
      image_dir:=/path/to/images fps:=10.0
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """构造 demo launch。"""
    share = get_package_share_directory("underwater_perception")
    config_dir = os.path.join(share, "config")

    image_dir = LaunchConfiguration("image_dir")
    fps = LaunchConfiguration("fps")
    settings_dir = LaunchConfiguration("settings_dir")
    settings_name = LaunchConfiguration("settings_name")
    rviz = LaunchConfiguration("rviz")

    declared = [
        DeclareLaunchArgument(
            "image_dir",
            default_value=os.path.join(share, "data", "euroc_sample"),
            description="按文件名排序的图片目录（需有运动，单目才能初始化）",
        ),
        DeclareLaunchArgument("fps", default_value="10.0"),
        DeclareLaunchArgument(
            "settings_dir",
            default_value=os.path.join(config_dir, "orbslam3") + "/",
        ),
        DeclareLaunchArgument("settings_name", default_value="EuRoC"),
        DeclareLaunchArgument("rviz", default_value="true"),
    ]

    player = Node(
        package="underwater_perception",
        executable="image_sequence_player",
        name="image_sequence_player",
        output="screen",
        parameters=[
            {"topic": "/image_raw", "synthetic": False, "image_dir": image_dir,
             "fps": ParameterValue(fps, value_type=float), "loop": True,
             "frame_id": "camera_optical"},
        ],
    )

    enhancer = Node(
        package="image_enhancer_cpp",
        executable="enhancer",
        name="image_enhancer_cpp",
        output="screen",
        parameters=[
            os.path.join(config_dir, "enhancer_underwater.yaml"),
            {"input_topic": "/image_raw", "output_topic": "/image_enhanced"},
        ],
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "slam_pipeline.launch.py")),
        launch_arguments={
            "start_slam": "true",
            "rviz": rviz,
            "settings_dir": settings_dir,
            "settings_name": settings_name,
        }.items(),
    )

    return LaunchDescription(declared + [player, enhancer, slam])
