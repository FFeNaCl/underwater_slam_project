"""一键验证链路（**不需要相机、不需要硬件**）。

这是给"在 Linux 上调代码的 AI/同学"准备的最短路径：一条命令把
图像源 -> 增强 -> 几何识别 -> SLAM 桥接 -> 语义地标 全部拉起来，
再按提示 `ros2 topic echo` 就能判断每一环是否正常工作。

用法::

    ros2 launch underwater_perception full_demo.launch.py
    ros2 launch underwater_perception full_demo.launch.py with_slam:=true rviz:=true

期望现象（按顺序检查，出问题就能定位到具体环节）
------------------------------------------------
1. ``ros2 topic hz /image_raw``            约 10 Hz
2. ``ros2 topic hz /image_enhanced``       与上面同频
3. ``ros2 topic echo /detections --once``  有 1 个 cube，conf > 0.5，带 track_id
4. ``ros2 topic hz /mono_py_driver/img_msg`` 与 /image_enhanced 同频（说明桥接在工作）
5. 若 ``with_slam:=true``：``ros2 topic echo /slam/status --once``
   —— 注意：**合成画面的纹理很少，ORB-SLAM3 很可能停在 NOT_INITIALIZED**，
   这是预期现象（它需要真实场景的纹理）。合成模式用来验证 1~4 环，SLAM 环节请用真实录像。
6. ``ros2 topic echo /semantic/landmarks --once`` 若没有 SLAM 位姿，会看到
   ``source: scale_fallback``（相机系近似，不代表地图尺度）；有 SLAM 位姿后变成 ``triangulated``。
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    """构造 launch 描述。"""
    share = get_package_share_directory("underwater_perception")
    orb_share = get_package_share_directory("ros2_orb_slam3")

    declared = [
        DeclareLaunchArgument("with_slam", default_value="false", description="是否连同 ORB-SLAM3 一起启动"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("player_fps", default_value="10.0"),
        DeclareLaunchArgument(
            "voc_file",
            default_value=os.path.join(orb_share, "orb_slam3", "Vocabulary", "ORBvoc.txt.bin"),
        ),
        DeclareLaunchArgument("settings_dir", default_value=os.path.join(share, "config", "orbslam3") + "/"),
        DeclareLaunchArgument("settings_name", default_value="water_1280x480"),
    ]

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "underwater_pipeline.launch.py")),
        launch_arguments={
            "image_source": "synthetic",
            "player_fps": LaunchConfiguration("player_fps"),
            "rviz": LaunchConfiguration("rviz"),
        }.items(),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "slam_pipeline.launch.py")),
        launch_arguments={
            "start_slam": LaunchConfiguration("with_slam"),
            "rviz": "false",
            "voc_file": LaunchConfiguration("voc_file"),
            "settings_dir": LaunchConfiguration("settings_dir"),
            "settings_name": LaunchConfiguration("settings_name"),
        }.items(),
    )

    return LaunchDescription(declared + [perception, slam])
