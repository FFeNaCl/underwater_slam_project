"""真实 USB 相机 launch：OpenCV 取流 -> ``/image_raw``。

```bash
ros2 launch underwater_perception camera.launch.py \
  camera_index:=0 width:=1280 height:=720 fourcc:=MJPG fps:=30.0
```

然后另开终端启动增强 + 检测::

    ros2 launch underwater_perception yolo_real.launch.py \
      model_path:=/home/hu/underwater_slam_project/models/best.pt

注意：``camera_index`` 对应 ``/dev/video0`` -> 0；``/dev/video1``、``/dev/video3``
通常是 UVC metadata 节点，打不开。
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """构造 launch 描述。"""
    camera_index = LaunchConfiguration("camera_index")
    video_device = LaunchConfiguration("video_device")
    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    fourcc = LaunchConfiguration("fourcc")
    fps = LaunchConfiguration("fps")
    topic = LaunchConfiguration("topic")
    frame_id = LaunchConfiguration("frame_id")

    declared = [
        DeclareLaunchArgument("camera_index", default_value="0", description="/dev/video0 -> 0"),
        DeclareLaunchArgument(
            "video_device",
            default_value="",
            description="稳定设备路径（优先于 camera_index），例如 /dev/v4l/by-id/...-video-index0",
        ),
        DeclareLaunchArgument("width", default_value="1280"),
        DeclareLaunchArgument("height", default_value="720"),
        DeclareLaunchArgument("fourcc", default_value="MJPG", description="MJPG 才有 30 fps"),
        DeclareLaunchArgument("fps", default_value="30.0"),
        DeclareLaunchArgument("topic", default_value="/image_raw"),
        DeclareLaunchArgument("frame_id", default_value="camera_optical"),
    ]

    camera_node = Node(
        package="underwater_perception",
        executable="camera",
        name="camera",
        output="screen",
        parameters=[
            {"camera_index": ParameterValue(camera_index, value_type=int)},
            {"device": video_device},
            {"width": ParameterValue(width, value_type=int)},
            {"height": ParameterValue(height, value_type=int)},
            {"fourcc": fourcc},
            {"fps": ParameterValue(fps, value_type=float)},
            {"topic": topic},
            {"frame_id": frame_id},
        ],
    )

    return LaunchDescription(declared + [camera_node])
