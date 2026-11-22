"""一键真实单目 SLAM 水下建图链路。

链路::

    真实相机 -> /image_raw
      -> image_enhancer_cpp -> /image_enhanced
      -> yolo_detector (best.pt) -> /yolo/detections + /yolo/annotated
      -> slam_input_bridge -> /mono_py_driver/{img_msg,timestep_msg}
      -> ORB-SLAM3 mono_node_cpp -> /slam/pose /slam/path /slam/map_points /slam/status
      -> semantic_landmark -> /semantic/landmarks + /semantic/markers
      -> RViz2

用法::

    source /opt/ros/humble/setup.bash
    source ~/underwater_slam_project/install/setup.bash

    ros2 launch underwater_perception mapping.launch.py

    # 只跑 SLAM，不开 YOLO
    ros2 launch underwater_perception mapping.launch.py use_yolo:=false

    # 无显示器/远程调试
    ros2 launch underwater_perception mapping.launch.py rviz:=false

注意：
* 相机移动后 ORB-SLAM3 才会初始化（单目纯旋转无法三角化）；
* `settings_name` 对应的 yaml 必须在 `settings_dir` 下，且分辨率与相机一致；
* 当前默认使用 `config/orbslam3/water_1280x480.yaml`（1280x720）。
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
    """构造一键建图 launch。"""
    share = get_package_share_directory("underwater_perception")
    orb_share = get_package_share_directory("ros2_orb_slam3")
    config_dir = os.path.join(share, "config")

    camera_index = LaunchConfiguration("camera_index")
    camera_device = LaunchConfiguration("camera_device")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_fourcc = LaunchConfiguration("camera_fourcc")
    camera_fps = LaunchConfiguration("camera_fps")
    use_yolo = LaunchConfiguration("use_yolo")
    use_landmark = LaunchConfiguration("use_landmark")
    model_path = LaunchConfiguration("model_path")
    detections_topic = LaunchConfiguration("detections_topic")
    voc_file = LaunchConfiguration("voc_file")
    settings_dir = LaunchConfiguration("settings_dir")
    settings_name = LaunchConfiguration("settings_name")
    rviz = LaunchConfiguration("rviz")
    viewer = LaunchConfiguration("viewer")
    start_slam = LaunchConfiguration("start_slam")
    enable_pangolin = LaunchConfiguration("enable_pangolin")
    point_publish_period_ms = LaunchConfiguration("point_publish_period_ms")

    declared = [
        DeclareLaunchArgument("camera_index", default_value="0", description="0=/dev/video0，2=第二台相机"),
        DeclareLaunchArgument(
            "camera_device",
            default_value="",
            description="稳定设备路径（优先于 camera_index），UVC 重新枚举后推荐使用",
        ),
        DeclareLaunchArgument("camera_width", default_value="1280"),
        DeclareLaunchArgument("camera_height", default_value="720"),
        DeclareLaunchArgument("camera_fourcc", default_value="MJPG"),
        DeclareLaunchArgument("camera_fps", default_value="30.0"),
        DeclareLaunchArgument("use_yolo", default_value="true"),
        DeclareLaunchArgument(
            "use_landmark",
            default_value="true",
            description="是否启动语义地标节点；纯几何建图可设 false",
        ),
        DeclareLaunchArgument(
            "model_path", default_value="/home/hu/underwater_slam_project/models/best.pt"
        ),
        DeclareLaunchArgument("detections_topic", default_value="/yolo/detections"),
        DeclareLaunchArgument(
            "voc_file",
            default_value=os.path.join(orb_share, "orb_slam3", "Vocabulary", "ORBvoc.txt.bin"),
        ),
        DeclareLaunchArgument(
            "settings_dir", default_value=os.path.join(config_dir, "orbslam3") + "/"
        ),
        DeclareLaunchArgument("settings_name", default_value="water_1280x480"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("viewer", default_value="false", description="是否同时打开 OpenCV 2D 俯视图"),
        DeclareLaunchArgument("start_slam", default_value="true"),
        DeclareLaunchArgument("enable_pangolin", default_value="false"),
        DeclareLaunchArgument("point_publish_period_ms", default_value="500"),
    ]

    camera_node = Node(
        package="underwater_perception",
        executable="camera",
        name="camera",
        output="screen",
        parameters=[
            {"camera_index": ParameterValue(camera_index, value_type=int)},
            {"device": camera_device},
            {"width": ParameterValue(camera_width, value_type=int)},
            {"height": ParameterValue(camera_height, value_type=int)},
            {"fourcc": camera_fourcc},
            {"fps": ParameterValue(camera_fps, value_type=float)},
            {"topic": "/image_raw"},
            {"frame_id": "camera_optical"},
        ],
    )

    enhancer_node = Node(
        package="image_enhancer_cpp",
        executable="enhancer",
        name="image_enhancer_cpp",
        output="screen",
        parameters=[
            os.path.join(config_dir, "enhancer_underwater.yaml"),
            {"input_topic": "/image_raw", "output_topic": "/image_enhanced"},
        ],
    )

    yolo_node = Node(
        package="yolo_vrx_bridge",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen",
        condition=IfCondition(use_yolo),
        parameters=[
            {
                "input_topic": "/image_enhanced",
                "annotated_topic": "/yolo/annotated",
                "detections_topic": detections_topic,
                "model_path": model_path,
                "require_model": True,
                "allow_label_replay": False,
                "label_topic": "",
                "dataset_root": "",
                "confidence_threshold": 0.25,
                "imgsz": 640,
                "device": "",
                "half": False,
                "max_det": 100,
                "frame_title": "Underwater YOLO mapping",
            }
        ],
    )

    bridge_node = Node(
        package="underwater_perception",
        executable="slam_input_bridge",
        name="slam_input_bridge",
        output="screen",
        parameters=[
            {"config_file": os.path.join(config_dir, "slam_bridge.yaml")},
            {"settings_name": settings_name},
        ],
    )

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
            {"enable_pangolin": ParameterValue(enable_pangolin, value_type=bool)},
            {"enable_opencv_window": False},
            {"publish_pose": True},
            {"publish_path": True},
            {"publish_points": True},
            {"publish_tf": True},
            {"publish_status": True},
            {"map_frame_id": "map"},
            {"camera_frame_id": "camera_optical"},
            {"max_path_length": 5000},
            {"point_publish_period_ms": ParameterValue(point_publish_period_ms, value_type=int)},
        ],
    )

    landmark_node = Node(
        package="underwater_perception",
        executable="semantic_landmark",
        name="semantic_landmark",
        output="screen",
        condition=IfCondition(use_landmark),
        parameters=[
            {"config_file": os.path.join(config_dir, "semantic_landmark.yaml")},
            {"detections_topic": detections_topic},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(rviz),
        arguments=["-d", os.path.join(config_dir, "rviz", "slam.rviz")],
    )

    viewer_node = Node(
        package="underwater_perception",
        executable="slam_viewer",
        name="slam_viewer",
        output="screen",
        condition=IfCondition(viewer),
    )

    return LaunchDescription(
        declared
        + [
            camera_node,
            enhancer_node,
            yolo_node,
            bridge_node,
            slam_node,
            landmark_node,
            rviz_node,
            viewer_node,
        ]
    )
