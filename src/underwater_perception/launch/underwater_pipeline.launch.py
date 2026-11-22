"""感知链路 launch：图像源 -> 图像增强 -> 几何旁路立方体识别。

用法::

    # 1) 真实相机主线（v4l2_camera）
    ros2 launch underwater_perception underwater_pipeline.launch.py image_source:=camera

    # 2) 无相机自测：合成水下场景驱动整条链路（Linux 侧 AI 验证用）
    ros2 launch underwater_perception underwater_pipeline.launch.py image_source:=synthetic

    # 3) 回放磁盘图片
    ros2 launch underwater_perception underwater_pipeline.launch.py image_source:=directory image_dir:=/path/to/images

    # 4) 数据集恢复后同时挂 YOLO
    ros2 launch underwater_perception underwater_pipeline.launch.py use_yolo:=true model_path:=/path/best.onnx

设计说明（改动前请先读）
------------------------
* ``image_source`` 是**单一**参数，三种取值互斥。早先用两个布尔参数（相机开关 + 播放器开关）
  会出现"相机与播放器同时向 /image_raw 发布"的竞争，画面帧率会莫名抖动，所以改成互斥设计。
* 所有 ``LaunchConfiguration`` 引用的参数都必须在**同一个** ``DeclareLaunchArgument`` 列表里先声明，
  否则 launch 执行到该 Node 时会因为取不到参数而失败（顺序敏感）。
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """构造 launch 描述。"""
    share = get_package_share_directory("underwater_perception")
    config_dir = os.path.join(share, "config")

    image_source = LaunchConfiguration("image_source")
    is_camera = PythonExpression(["'", image_source, "' == 'camera'"])
    is_player = PythonExpression(["'", image_source, "' != 'camera'"])
    is_synthetic = PythonExpression(["'", image_source, "' == 'synthetic'"])

    declared = [
        DeclareLaunchArgument(
            "image_source",
            default_value="camera",
            choices=["camera", "synthetic", "directory"],
            description="图像来源：camera（v4l2_camera）/ synthetic（合成场景）/ directory（回放图片）",
        ),
        DeclareLaunchArgument("image_dir", default_value="", description="image_source:=directory 时的图片目录"),
        DeclareLaunchArgument("video_device", default_value="/dev/video0", description="image_source:=camera 时的设备"),
        DeclareLaunchArgument("player_fps", default_value="10.0", description="合成/回放的发布帧率"),
        DeclareLaunchArgument("use_yolo", default_value="false", description="是否同时启动 YOLO 检测节点"),
        DeclareLaunchArgument("model_path", default_value="", description="YOLO ONNX 权重路径"),
        DeclareLaunchArgument("rviz", default_value="false", description="是否启动 RViz2"),
        DeclareLaunchArgument("cube_config", default_value=os.path.join(config_dir, "cube_detector.yaml")),
        DeclareLaunchArgument(
            "enhancer_params", default_value=os.path.join(config_dir, "enhancer_underwater.yaml")
        ),
    ]

    # 1) 图像源（互斥）
    camera_node = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="v4l2_camera",
        output="screen",
        condition=IfCondition(is_camera),
        parameters=[{"video_device": LaunchConfiguration("video_device"), "image_size": [1280, 480]}],
        remappings=[("image_raw", "/image_raw")],
    )

    player_node = Node(
        package="underwater_perception",
        executable="image_sequence_player",
        name="image_sequence_player",
        output="screen",
        condition=IfCondition(is_player),
        parameters=[
            {
                "topic": "/image_raw",
                "synthetic": is_synthetic,
                "image_dir": LaunchConfiguration("image_dir"),
                "fps": LaunchConfiguration("player_fps"),
                "width": 1280,
                "height": 480,
                "frame_id": "camera_optical",
            }
        ],
    )

    # 2) 水下图像增强（既有 C++ 节点）
    #    注意：真实相机主线里它订阅 /image_raw 并发布 /image_enhanced；
    #    标定完成后把配置里的 use_undistort 打开即可同时去畸变。
    enhancer_node = Node(
        package="image_enhancer_cpp",
        executable="enhancer",
        name="image_enhancer_cpp",
        output="screen",
        parameters=[LaunchConfiguration("enhancer_params")],
    )

    # 3) 几何旁路识别
    detector_node = Node(
        package="underwater_perception",
        executable="cube_detector",
        name="cube_detector",
        output="screen",
        parameters=[{"config_file": LaunchConfiguration("cube_config")}],
    )

    # 4) 可选：YOLO（发布到 /yolo/detections，与几何旁路共用同一 JSON 契约）
    yolo_node = Node(
        package="yolo_vrx_bridge",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_yolo")),
        parameters=[
            {
                "input_topic": "/image_enhanced",
                "model_path": LaunchConfiguration("model_path"),
                "annotated_topic": "/yolo/annotated",
                "detections_topic": "/yolo/detections",
                "allow_label_replay": False,
            }
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

    return LaunchDescription(declared + [camera_node, player_node, enhancer_node, detector_node, yolo_node, rviz_node])
