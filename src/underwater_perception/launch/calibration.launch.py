"""标定 launch：启动标定节点（边采图边标定）。

用法::

    ros2 launch underwater_perception calibration.launch.py image_topic:=/image_enhanced
    # 另开终端：观察 /calibration/debug 画面，够了以后执行
    ros2 service call /calibration_node/run std_srvs/srv/Trigger "{}"

提示
----
* 标定前建议先把图像增强跑起来（水下标定要用**增强后**的图像，与实际使用一致）。
* 采集期间让标定板出现在画面四角与不同倾角，覆盖度不足会在报告里被明确警告。
* 输出目录默认在启动目录下的 ``calibration_out``，可用 ``out_dir`` 参数指定绝对路径。
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
    config_dir = os.path.join(share, "config")

    declared = [
        DeclareLaunchArgument("image_topic", default_value="/image_enhanced", description="标定使用的图像话题"),
        DeclareLaunchArgument("config_file", default_value=os.path.join(config_dir, "calibration.yaml")),
        DeclareLaunchArgument("out_dir", default_value="calibration_out", description="输出目录（建议给绝对路径）"),
        DeclareLaunchArgument("auto_capture", default_value="true", description="是否自动周期采集"),
        DeclareLaunchArgument("capture_interval", default_value="1.0", description="自动采集的最小间隔（秒）"),
        DeclareLaunchArgument("rviz", default_value="false", description="是否顺带启动 rqt/rviz 观察（rviz 仅显示调试图）"),
    ]

    calibration_node = Node(
        package="underwater_perception",
        executable="calibration_node",
        name="calibration_node",
        output="screen",
        parameters=[
            {"config_file": LaunchConfiguration("config_file")},
            {"image_topic": LaunchConfiguration("image_topic")},
            {"output_dir": LaunchConfiguration("out_dir")},
            {"auto_capture": LaunchConfiguration("auto_capture")},
            {"capture_interval_s": LaunchConfiguration("capture_interval")},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", os.path.join(config_dir, "rviz", "detection.rviz")],
    )

    return LaunchDescription(declared + [calibration_node, rviz_node])
