"""真实相机 YOLO 链路：``/image_raw`` -> 增强 -> ``best.pt`` 检测。

相机本身请用 ``v4l2_camera`` 单独启动（详见 Readme.md），避免和本 launch 抢设备::

    ros2 run v4l2_camera v4l2_camera_node --ros-args \
      -p video_device:=/dev/video0 -p image_size:="[1280, 720]" \
      -p pixel_format:=MJPG -r image_raw:=/image_raw

然后::

    ros2 launch underwater_perception yolo_real.launch.py \
      model_path:=/home/hu/underwater_slam_project/models/best.pt

``model_path`` 留空时节点会明确显示 "MODEL NOT LOADED"，不会回放标签冒充检测结果。
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """构造 launch 描述。"""
    share = get_package_share_directory("underwater_perception")
    config_dir = os.path.join(share, "config")

    camera_topic = LaunchConfiguration("camera_topic")
    detector_topic = LaunchConfiguration("detector_topic")
    use_enhancer = LaunchConfiguration("use_enhancer")
    model_path = LaunchConfiguration("model_path")
    confidence_threshold = LaunchConfiguration("confidence_threshold")
    imgsz = LaunchConfiguration("imgsz")
    device = LaunchConfiguration("device")
    half = LaunchConfiguration("half")
    max_det = LaunchConfiguration("max_det")

    declared = [
        DeclareLaunchArgument("camera_topic", default_value="/image_raw", description="相机原图话题"),
        DeclareLaunchArgument("detector_topic", default_value="/image_enhanced", description="检测输入话题"),
        DeclareLaunchArgument("use_enhancer", default_value="true", description="是否启动水下图像增强"),
        DeclareLaunchArgument(
            "model_path",
            default_value="",
            description="训练好的 best.pt / best.onnx 绝对路径（留空 = 显示未加载警告）",
        ),
        DeclareLaunchArgument("confidence_threshold", default_value="0.25"),
        DeclareLaunchArgument("imgsz", default_value="640"),
        DeclareLaunchArgument("device", default_value="", description='""=自动，cpu 或 0'),
        DeclareLaunchArgument("half", default_value="false"),
        DeclareLaunchArgument("max_det", default_value="100"),
    ]

    enhancer_node = Node(
        package="image_enhancer_cpp",
        executable="enhancer",
        name="image_enhancer_cpp",
        output="screen",
        condition=IfCondition(use_enhancer),
        parameters=[
            os.path.join(config_dir, "enhancer_underwater.yaml"),
            {"input_topic": camera_topic, "output_topic": detector_topic},
        ],
    )

    detector_node = Node(
        package="yolo_vrx_bridge",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen",
        parameters=[
            {
                "input_topic": detector_topic,
                "annotated_topic": "/yolo/annotated",
                "detections_topic": "/yolo/detections",
                "model_path": model_path,
                "confidence_threshold": ParameterValue(confidence_threshold, value_type=float),
                "imgsz": ParameterValue(imgsz, value_type=int),
                "device": device,
                "half": ParameterValue(half, value_type=bool),
                "max_det": ParameterValue(max_det, value_type=int),
                "require_model": True,
                "allow_label_replay": False,
                "label_topic": "",
                "dataset_root": "",
                "frame_title": "Underwater YOLO (real camera)",
            }
        ],
    )

    return LaunchDescription(declared + [enhancer_node, detector_node])
