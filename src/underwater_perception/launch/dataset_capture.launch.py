"""真实相机 YOLO 数据集采集 launch。

前置：相机已在发布 ``/image_raw``，增强节点可在发布 ``/image_enhanced``::

    ros2 launch underwater_perception dataset_capture.launch.py \
      image_topic:=/image_enhanced \
      output_dir:=/home/hu/underwater_slam_project/datasets/real_underwater/raw \
      save_interval:=0.5 max_images:=600

停止/补拍::

    ros2 service call /dataset_capture/stop std_srvs/srv/Trigger "{}"
    ros2 service call /dataset_capture/snapshot std_srvs/srv/Trigger "{}"
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """构造 launch 描述。"""
    share = get_package_share_directory("underwater_perception")
    config_dir = os.path.join(share, "config")

    config_file = LaunchConfiguration("config_file")
    image_topic = LaunchConfiguration("image_topic")
    output_dir = LaunchConfiguration("output_dir")
    save_interval = LaunchConfiguration("save_interval")
    max_images = LaunchConfiguration("max_images")
    write_empty_labels = LaunchConfiguration("write_empty_labels")

    declared = [
        DeclareLaunchArgument("config_file", default_value=os.path.join(config_dir, "dataset_capture.yaml")),
        DeclareLaunchArgument("image_topic", default_value="/image_enhanced"),
        DeclareLaunchArgument(
            "output_dir", default_value="/home/hu/underwater_slam_project/datasets/real_underwater/raw"
        ),
        DeclareLaunchArgument("save_interval", default_value="0.5"),
        DeclareLaunchArgument("max_images", default_value="0", description="0 = 不限"),
        DeclareLaunchArgument("write_empty_labels", default_value="false"),
    ]

    capture_node = Node(
        package="underwater_perception",
        executable="dataset_capture",
        name="dataset_capture",
        output="screen",
        parameters=[
            {"config_file": config_file},
            {"image_topic": image_topic},
            {"output_dir": output_dir},
            {"save_interval_s": ParameterValue(save_interval, value_type=float)},
            {"max_images": ParameterValue(max_images, value_type=int)},
            {"write_empty_labels": ParameterValue(write_empty_labels, value_type=bool)},
        ],
    )

    return LaunchDescription(declared + [capture_node])
