# 未验证项清单（UNVERIFIED）——**请勿跳过或误读**

本文件的目的：把"已经真实测过"和"只是写好了"严格分开，避免把未验证的东西当成结论
写进报告或答辩。每完成一项验证，请把状态改成 ✅ 并补上实测数据与日期。

开发环境事实：开发机为 **Windows，无 ROS2、无相机、无水下环境**。
因此下列分组是硬边界。

---

## A. 已真实验证（合成数据 + pytest，可直接引用）

| 项 | 结论 | 证据 |
|---|---|---|
| 几何检测（多视角） | 6/6 检出，IoU 均值 0.968 / 最小 0.933 | `selftest_out/selftest_report.json` |
| 顶点数判断 | 正视 4、斜视 6，全部正确 | 同上 `vertex_counts` |
| 单帧距离（已知边长 5 cm） | 误差 ≤0.90% | 同上 `max_distance_rel_error` |
| 误检抑制 | 背景 0、蓝色立方体 0、严格模式红色平面方块 0 | 同上 `false_positives` |
| 跟踪稳定性 | 10 帧 9 次发布、单 ID | 同上 `tracking_stability` |
| 标定内参恢复 | fx/fy 误差 0.80%/0.81%，主点 2.4 px，RMS 0.404 px | 同上 `calibration_math` |
| 三角化地标 | 无噪声 ~1e-14 m；1 px 噪声 < 5 cm | `test/test_landmarks.py` |
| 消息契约 | JSON 自洽且为既有 YOLO 契约超集 | `test/test_types.py` |
| 配置/launch/打包一致性 | 99 项 pytest 全绿（含键名、launch 参数声明、入口名与 launch `executable` 对应、`%YAML:1.0` 首行、数据集工具） | `pytest test -q` |
| 两个工具 CLI | 合成检测 6/6；标定 16 视图 RMS 0.404 px 并写出四件套 | 运行输出 |
| 标定 GUI | Tk 窗口构建、标定板参数摘要、日志输出正常 | 本机 `DISPLAY=:0` 冒烟测试 |
| 数据集采集/检查 | 视频抽帧、session JSON、data.yaml；缺失标签/非法行/跨集重复/孤立标签 | `test/test_dataset_tools.py` |
| YOLO 节点真实配置 | 空 `model_path` + `require_model=true` 启动不崩溃、明确报警；空 `label_topic` 不建订阅 | 本机 `ros2 run yolo_vrx_bridge yolo_detector` |
| ONNX 推理 | `models/best.onnx` 在空白图上推理成功（0 框） | ultralytics 8.4.50 |
| Python 包构建 | `underwater_perception`、`yolo_vrx_bridge` 普通 `colcon build` 通过 | 本机 2026-11-07 |
| C++ 增强节点构建 | `image_enhancer_cpp` `colcon build` 通过（6.7 s） | 本机 2026-11-07 |
| launch 参数 | `yolo_real.launch.py`、`dataset_capture.launch.py` 的 `-s` 参数列表正常 | 本机 |
| 真实 YOLO 链路启动 | `yolo_real.launch.py` 同时拉起 enhancer + yolo_detector；空模型时明确报错不崩溃 | 本机（无相机） |
| 真实相机取流 | OpenCV：YUYV 1280x720 10 fps、MJPG 1280x720 22 fps、YUYV 640x480 28 fps；`v4l2_camera` 640x480 约 10 fps；`usb_cam` 1280x720 约 2 fps；新增 `camera` 节点 640x480 约 19 fps | 本机 2026-11-07 |
| UVC metadata 节点 | `/dev/video1` 无格式、OpenCV 打不开；video0/video2 才是取流节点 | `v4l2-ctl --list-formats-ext` |
| ORB-SLAM3 C++ 编译 | `ros2_orb_slam3` 编译通过（2 分 07 秒） | `colcon build --packages-select ros2_orb_slam3` |
| ORB-SLAM3 端到端（EuRoC 序列） | `tracking_state=OK`、4464 地图点、`/slam/pose` 9.9 Hz、`/slam/path` 10 Hz、`/slam/map_points` 0.93 Hz | `slam_demo.launch.py` |
| RViz 配置 | `slam.rviz`/`detection.rviz` 为合法 YAML，RViz 窗口正常创建（1600x900） | 本机 |
| OpenCV 俯视可视化 | `slam_viewer` 窗口显示轨迹/地图点/状态（实测 2042 点、167 位姿） | 本机 |

## B. 完全未验证（按风险排序，**必须在 Linux 上做**）

| # | 项 | 风险 | 验证命令 | 通过标准 | 状态 |
|---|---|---|---|---|---|
| B1 | C++ 补丁能否编译 | **高** | `cd ~/underwater_slam_project && colcon build` | 无编译错误 | ✅ 2026-11-19，2m07s |
| B2 | SLAM 节点能否发布位姿/点云 | 高 | `ros2 topic hz /slam/pose`、`ros2 topic hz /slam/map_points` | 有稳定频率 | ✅ EuRoC 序列实测 9.9 Hz / 0.93 Hz |
| B3 | ROS 节点能否启动（导入/参数/QoS） | 中 | `ros2 launch underwater_perception mapping.launch.py rviz:=false` | 无异常退出 | ✅ 真实相机链路已启动 |
| B4 | launch 在 Humble 的兼容性 | 中 | 同上 + `ros2 launch ... -s` | 参数齐全 | ✅ `mapping`/`slam_demo` `-s` 通过 |
| B5 | 桥接与 C++ 的握手 | 中 | `ros2 topic echo /mono_py_driver/exp_settings_ack --once` | 看到 `ACK` | ✅ EuRoC/真实相机均收到 ACK |
| B6 | `TransformBroadcaster` 构造重载 | 中 | 编译期若报错，按 `SLAM_PUBLISHERS.md §9` 改 | 编译通过 + `ros2 run tf2_tools view_frames` | ✅ 编译通过 |
| B7 | 真实相机图像的检出率 | **未知** | 采 50~100 张真实图，`cube_detect_cli --input <dir>` | 记录检出率/误检数 | ⬜ |
| B8 | 真实水下 HSV 阈值 | **未知** | `--print-mask` 看掩码后调 `config/cube_detector.yaml` | 掩码稳定覆盖目标 | ⬜ |
| B9 | 水下标定实际精度 | 未知 | 按 `CALIBRATION.md` 实测 | RMS<0.5 px 且去畸变后直线变直 | ⬜ |
| B10 | 单帧测距 vs 卷尺实测 | 未知 | 固定距离拍 10 张，比较 `measurement.distance_m` | 记录实际误差分布 | ⬜ |
| B11 | SLAM 在水下的初始化与漂移 | 未知 | 水池短序列跑通，记录丢失次数与回环漂移 | 能初始化、轨迹连续 | ⬜ |
| B12 | 语义地标与点云尺度一致性 | 未知 | 位姿可用后看 `/semantic/landmarks` 的 `source` | 出现 `triangulated` 且与点云重合 | ⬜ |
| B13 | 合成画面能否让 ORB-SLAM3 初始化 | 预期失败 | `full_demo.launch.py with_slam:=true` | 若停在 `NOT_INITIALIZED` 属**正常** | ⬜ |
| B14 | 真实相机 ROS 采集（`/image_enhanced` -> 数据集） | 中 | `ros2 launch underwater_perception dataset_capture.launch.py` | 连续保存、`capture_session.json` 计数一致 | ⬜ |
| B15 | 真实环境 `best.pt` 推理 | 高 | 填 `model_path` 后跑 `yolo_real.launch.py` | `/yolo/detections` 的 `source=model`、框合理 | ⬜ |
| B16 | 标定 GUI 在真实相机上的交互采集 | 低 | `ros2 run underwater_perception calibrate_gui` | SPACE 采集、C 标定、四件套写出 | ⬜ |

## C. 已知限制（不是 bug，但必须在报告里说明）

1. **单目无绝对尺度**：SLAM 地图单位不是米；只有"已知边长测距"或外部尺度来源才有物理尺度。
2. **单帧朝向不可唯一观测**：立方体是无纹理对称物体，`pose_rvec` 只到对称等价类，
   不要用它做精细姿态结论。
3. **正视单面视角与平面红色方块原理上不可区分**（只有 4 顶点、无内部棱线）；
   严格模式会拒掉这类视角。
4. **水下折射**：平面防水壳的多介质折射使针孔+径向畸变只是近似，边缘误差偏大。
5. **合成棋盘格用直线近似畸变曲线边**，因此标定恢复出的高阶畸变系数（k2/k3）
   与真值有系统性偏差（实测 k1 误差 0.019、k2 误差 0.060）——这不代表真实数据上的表现。
6. **YOLO 数据集未就绪**：几何旁路是临时方案，等数据齐了应做对比实验。
7. **`GetAllMapPoints()` 的裸指针**存在极小悬垂窗口（与上游官方 wrapper 相同），补丁里已判空。

## D. 验证完成后请更新
1. 把 B 表对应行状态改为 ✅，并写上**实测数值**（不要只写"通过"）。
2. 把新增的数字回填到 `docs/AI_HANDOFF.md` 第 0 节的基线表（合成 vs 真实分开列）。
3. 若真实数据上的 HSV 阈值与默认值差异较大，请把新值写进 `config/cube_detector.yaml`
   的注释里，并注明"实测于 <日期/水质/距离>"。
4. 若发现新陷阱，追加到 `AI_HANDOFF.md` 第 8 节（那张表就是为了不再踩第二次）。
