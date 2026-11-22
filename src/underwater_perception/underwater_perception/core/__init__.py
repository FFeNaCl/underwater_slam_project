"""纯算法层：不 import rclpy，因此可在无 ROS 环境（含 Windows 开发机）单测。

对外稳定入口：
    types.py          检测消息数据契约（schema_version=1），与现有 yolo 节点兼容
    params.py         扁平参数模型 + YAML 加载（ROS 参数文件与 CLI 共用同一份 YAML）
    colors.py         颜色分割（HSV 预设 + 红色通道优势校验）
    cube_geometry.py  几何旁路立方体检测（分割 -> 轮廓 -> 几何验证 -> 打分）
    tracker.py        跨帧 IoU 跟踪，给语义地标提供稳定 track_id
    measurement.py    单目测距/方位/射线三角化（已知边长或 SLAM 位姿）
    landmarks.py      多视图语义地标管理（三角化 + 地图点维护）
    calibration.py    标定数学与四种输出文件（ost / ORB-SLAM3 / 增强节点 / 报告）
    synthetic.py      合成水下场景与合成棋盘格：无硬件即可端到端自测
    cv_compat.py      OpenCV 4.x / 5.x 差异兼容（本机 cv2 为 5.0.0，Ubuntu 为 4.5.4）
"""
