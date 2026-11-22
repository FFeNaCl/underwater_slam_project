"""SLAM 链路 launch：图像增强 -> SLAM 输入桥接 -> ORB-SLAM3 -> 语义地标 -> RViz。

用法::

    # 1) 先确认设置文件与词袋模型路径正确（下面两个参数是关键）
    ros2 launch underwater_perception slam_pipeline.launch.py \\
      voc_file:=/home/hu/underwater_slam_project/models/ORBvoc.txt.bin \\
      settings_dir:=/home/hu/underwater_slam_project/config/orbslam3 \\
      settings_name:=water_1280x480 rviz:=true

    # 2) 只启动桥接 + 语义地标（SLAM 节点你自己在别的终端起）
    ros2 launch underwater_perception slam_pipeline.launch.py start_slam:=false

关键点（踩过坑的地方）
----------------------
* ORB-SLAM3 的 C++ 节点把设置文件名拼成 ``<settings_dir>/<settings_name>.yaml``，
  所以 ``settings_dir`` **必须以 / 结尾或用文件名拼接后仍正确**，且该 yaml 必须真实存在。
* 词袋模型 ``ORBvoc.txt.bin`` 约 48 MB，不在本项目仓库里，需要指向实际文件
  （仓库自带于 ``ros2_orb_slam3/orb_slam3/Vocabulary/``）。
* C++ 节点默认 **不开 Pangolin 窗口**（``enable_pangolin=false``），无显示器也能跑；
  需要可视化时加 ``-p enable_pangolin:=true``（要求机器有图形界面）。
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """构造 launch 描述。"""
    share = get_package_share_directory("underwater_perception")
    orb_share = get_package_share_directory("ros2_orb_slam3")
    config_dir = os.path.join(share, "config")

    bridge_config = LaunchConfiguration("bridge_config")
    landmark_config = LaunchConfiguration("landmark_config")
    start_slam = LaunchConfiguration("start_slam")
    rviz = LaunchConfiguration("rviz")
    voc_file = LaunchConfiguration("voc_file")
    settings_dir = LaunchConfiguration("settings_dir")
    settings_name = LaunchConfiguration("settings_name")
    enable_pangolin = LaunchConfiguration("enable_pangolin")

    declared = [
        DeclareLaunchArgument("bridge_config", default_value=os.path.join(config_dir, "slam_bridge.yaml")),
        DeclareLaunchArgument("landmark_config", default_value=os.path.join(config_dir, "semantic_landmark.yaml")),
        DeclareLaunchArgument("start_slam", default_value="true", description="是否同时启动 ORB-SLAM3 的 C++ 节点"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "voc_file",
            default_value=os.path.join(orb_share, "orb_slam3", "Vocabulary", "ORBvoc.txt.bin"),
            description="ORB 词袋模型路径（48 MB，必须真实存在）",
        ),
        DeclareLaunchArgument(
            "settings_dir",
            default_value=os.path.join(config_dir, "orbslam3") + "/",
            description="存放 <settings_name>.yaml 的目录（以 / 结尾最稳）",
        ),
        DeclareLaunchArgument("settings_name", default_value="water_1280x480"),
        DeclareLaunchArgument("enable_pangolin", default_value="false", description="是否开 ORB-SLAM3 的 Pangolin 窗口"),
    ]

    # 1) SLAM 输入桥接：把 /image_enhanced 喂给 ORB-SLAM3 期望的两个话题，并完成握手
    bridge_node = Node(
        package="underwater_perception",
        executable="slam_input_bridge",
        name="slam_input_bridge",
        output="screen",
        parameters=[
            {"config_file": bridge_config},
            {"settings_name": settings_name},
        ],
    )

    # 2) ORB-SLAM3 单目节点（已打补丁：发布 /slam/pose、/slam/path、/slam/map_points、/slam/status 与 TF）
    slam_node = Node(
        package="ros2_orb_slam3",
        executable="mono_node_cpp",
        name="mono_node_cpp",
        output="screen",
        condition=IfCondition(start_slam),
        parameters=[
            {"node_name_arg": "mono_slam_cpp"},
            {"voc_file_arg": voc_file},
            {"settings_file_path_arg": settings_dir},
            {"enable_pangolin": enable_pangolin},
            {"enable_opencv_window": False},
            {"publish_pose": True},
            {"publish_path": True},
            {"publish_points": True},
            {"publish_tf": True},
            {"map_frame_id": "map"},
            {"camera_frame_id": "camera"},
        ],
    )

    # 3) 语义地标：检测 + SLAM 位姿 -> 三角化地标
    landmark_node = Node(
        package="underwater_perception",
        executable="semantic_landmark",
        name="semantic_landmark",
        output="screen",
        parameters=[{"config_file": landmark_config}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(rviz),
        arguments=["-d", os.path.join(config_dir, "rviz", "slam.rviz")],
    )

    return LaunchDescription(declared + [bridge_node, slam_node, landmark_node, rviz_node])
