# AI 交接说明（AI_HANDOFF）

> **给在 Linux（Ubuntu 22.04 + ROS2 Humble）上接手本项目的 AI/同学：先读这一份。**
> 本文件由开发机（Windows，**没有 ROS2**）上的 AI 编写。因此：
> **`core/` 里的算法在 Windows 上用合成数据真实跑过并通过测试；ROS 节点、launch、
> C++ 补丁已在 Linux 上编译并跑通 EuRoC 序列（2026-11-19），但真实水下仍未验证。
> 两者在下文明确分开标注，请勿混淆。

---

## 0. 五分钟建立信任（先做这一步，别读代码）

在 Linux 工作区根目录（`~/underwater_slam_project`）执行：

```bash
cd ~/underwater_slam_project
python3 -m pip install --user numpy opencv-python pyyaml pytest    # 或复用 ROS 自带的 python3-opencv

# ① 纯算法自测：不需要 ROS、不需要相机、不需要数据集
PYTHONPATH=src/underwater_perception python3 -m underwater_perception.tools.selftest --out ./selftest_out
echo "退出码 $?"     # 0 = 全部通过；报告在 selftest_out/selftest_report.json

# ② 单元测试（99 项：消息契约、参数、检测、跟踪、几何、标定、地标、数据集工具、配置与打包静态检查）
cd src/underwater_perception && PYTHONPATH=. python3 -m pytest test -q

# ③ 编译 ROS 包
cd ~/underwater_slam_project && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

三条都通过，说明**算法层与代码一致性没问题**，接下来只是 ROS 集成调试（第 6 节有故障手册）。

### 开发机上实测到的数字（合成数据，可作为回归基线）

| 指标 | 实测值 | 来源 |
|---|---|---|
| 合成立方体检出率 | 6/6（100%） | selftest `geometry_detection` |
| 轮廓与真值 IoU | 均值 0.968，最小 0.933 | 同上 |
| 顶点数判断 | 正视 4 顶点 / 斜视 6 顶点，全部正确 | 同上 |
| 单帧距离估计误差（已知边长 5 cm） | 最大 0.90%，最好 0.08% | 同上（斜视也成立） |
| 误检抑制 | 背景 0、蓝色立方体 0、严格模式下红色平面方块 0 | selftest `false_positive_suppression` |
| 跟踪稳定性 | 10 帧序列 9 帧发布、track_id 恒定为 1 | selftest `tracking_stability` |
| 标定内参恢复误差 | fx/fy 0.80%/0.81%，主点 2.4 px | selftest `calibration_math` |
| 标定 RMS | 0.404 px（18 视图） | 同上 |
| 三角化地标误差 | 无噪声 ~1e-14 m；1 px 像素噪声 < 5 cm | pytest `test_landmarks.py` |

这些数字**只在合成数据上成立**，真实水下的预期差异见第 5 节。

---

## 1. 目录地图（精确路径）

```
~/underwater_slam_project/                      # 本项目工作区
├── Readme.md                              # 原始项目说明（历史，未改动）
├── 下一步工作Readme.md                     # 原始工作计划（历史，未改动）
├── AGENTS.md                              # 给 AI 的约定（本文件是详细版）
├── config/                                # 既有配置（未改动）
│   ├── enhancer_underwater.yaml           # 增强节点参数（权威版本）
│   ├── water_1280x480.orbslam3.example.yaml
│   └── yolo_vrx_demo.yaml
├── models/best.onnx                       # 既有 YOLOv8 权重
├── src/
│   ├── image_enhancer_cpp/                # 既有：水下图像增强（C++，未改动）
│   ├── yolo_vrx_bridge/                   # 既有：YOLO 检测 / 数据集回放（未改动）
│   └── underwater_perception/             # ★ 本次新增（本文件的重点）
│       ├── package.xml  setup.py  setup.cfg
│       ├── underwater_perception/
│       │   ├── core/                      # 纯算法层（无 ROS 依赖，已充分测试）
│       │   │   ├── types.py               #   检测消息契约（schema_version=1）
│       │   │   ├── params.py              #   扁平参数模型 + YAML 加载
│       │   │   ├── colors.py              #   HSV 分割 + 红色通道优势校验
│       │   │   ├── cube_geometry.py       #   几何旁路检测器（分割→轮廓→几何验证→打分）
│       │   │   ├── tracker.py             #   跨帧 IoU 跟踪
│       │   │   ├── measurement.py         #   单目测距/射线/三角化/立方体轮廓拟合
│       │   │   ├── landmarks.py           #   多视图语义地标（三角化 + 剔除 + 维护）
│       │   │   ├── calibration.py         #   标定数学 + 四件套输出
│       │   │   ├── synthetic.py           #   合成水下场景 / 合成棋盘格（自测基础设施）
│       │   │   ├── visualize.py           #   标注绘制（节点与 CLI 共用，保证画面一致）
│       │   │   └── cv_compat.py           #   OpenCV 4.x/5.x 兼容层
│       │   ├── nodes/                     # ROS2 薄封装（未在真机验证过！）
│       │   │   ├── image_bridge.py        #   ROS Image <-> numpy（不依赖 cv_bridge）
│       │   │   ├── cube_detector_node.py  #   /image_enhanced -> /detections
│       │   │   ├── calibration_node.py    #   边采图边标定（含 3 个 Trigger 服务）
│       │   │   ├── slam_input_bridge_node.py  # 图像 -> ORB-SLAM3 输入话题 + 握手
│       │   │   ├── camera_node.py            # ★ OpenCV MJPG 取流 -> /image_raw
│       │   │   ├── slam_viewer_node.py       # ★ OpenCV 2D 俯视图（轨迹/点云/地标）
│       │   │   ├── semantic_landmark_node.py  # 检测 + 位姿 -> 语义地标
│       │   │   ├── dataset_capture_node.py   # ★ ROS 图像按间隔落盘（数据集采集）
│       │   │   └── image_sequence_player_node.py # 合成/序列播放（无相机验证用）
│       │   └── tools/                     # 命令行工具（无 ROS 也能用）
│       │       ├── selftest.py            #   无硬件自测（★ 先跑这个）
│       │       ├── calibrate_cli.py       #   离线标定（图片/视频/相机/ROS 话题）
│       │       ├── calibrate_gui.py       #   ★ 图形标定（标定板参数可改）
│       │       ├── capture_dataset.py     #   ★ 真实相机/视频 YOLO 数据采集
│       │       ├── yolo_dataset_check.py  #   ★ 数据集标签/泄漏体检
│       │       └── cube_detect_cli.py     #   几何识别（合成/图片/视频）
│       ├── launch/                        # ★ mapping（真实相机一键建图）/ slam_demo（无相机验证）
│       │                                  #   + camera / dataset_capture / yolo_real / underwater_pipeline
│       │                                  #   + slam_pipeline / calibration / full_demo
│       ├── config/                        # cube_detector / calibration / slam_bridge /
│       │                                  # semantic_landmark / enhancer_underwater / orbslam3/ / rviz/
│       ├── test/                          # 99 项 pytest（可在无 ROS 环境运行）
│       └── docs/                          # 本目录（AI 文档集）
└── ../orb_slam3_ws/src/ros2_orb_slam3/    # 第三方 ORB-SLAM3 封装（本次打了补丁，见第 4.4 节）
    ├── SLAM_PUBLISHERS.md                 # 补丁的完整说明（★ 读它了解 C++ 改动）
    └── patches/ros2_orb_slam3_slam_publishers.patch   # 可回滚的 patch
```

---

## 2. 构建与运行环境

* Ubuntu 22.04 + ROS2 Humble；`image_enhancer_cpp`（既有 C++ 包）依赖 `cv_bridge`、`OpenCV`。
* Python 依赖（本项目）：`numpy`、`opencv-python`（ROS 已带 `python3-opencv`）、`pyyaml`。
  节点只用 `rclpy` + 标准消息（`sensor_msgs`/`std_msgs`/`std_srvs`/`geometry_msgs`/`nav_msgs`/`visualization_msgs`），
  **不需要** `vision_msgs`、`cv_bridge`（图像转换自己实现了，见 `nodes/image_bridge.py`）。
* `ros2_orb_slam3` 需要：Eigen3、Pangolin、OpenCV ≥ 4.2、C++17。补丁额外依赖
  `geometry_msgs`、`nav_msgs`、`tf2_ros`（已写进它的 `package.xml` / `CMakeLists.txt`）。
* **词袋模型** `ORBvoc.txt.bin`（约 48 MB）在
  `orb_slam3_ws/src/ros2_orb_slam3/orb_slam3/Vocabulary/` 下，启动 SLAM 时要用绝对路径指过去。

构建：

```bash
cd ~/underwater_slam_project && source /opt/ros/humble/setup.bash && colcon build --symlink-install
cd ~/underwater_slam_project && source /opt/ros/humble/setup.bash && colcon build --symlink-install
# 两个工作区都要 source：
source ~/underwater_slam_project/install/setup.bash && source ~/underwater_slam_project/install/setup.bash
```

---

## 3. 不可破坏的分层约定（改代码前必读）

1. **`core/` 不许 import rclpy。** 这是本项目能在没有 ROS 的机器上做真实测试的唯一原因。
   若需要新算法，先写成 `core/` 里的纯函数 + pytest，再在 `nodes/` 里包一层。
2. **`nodes/` 不许写几何/数学。** 节点里只应有：参数声明、消息转换、发布、日志。
3. **参数键名唯一来源是 `core/params.py`。** 配置文件、节点 `declare_parameter`、CLI 参数
   都从 `to_flat()` / `from_flat()` 派生。`test/test_configs.py` 会静态校验
   "YAML 里的键 ⊆ 代码认识的键"，键名写错会直接测试失败（而不是现场默默不生效）。
4. **消息契约见 `core/types.py` 与 `docs/INTERFACES.md`。**
   `/detections` 的 JSON 是既有 YOLO 节点输出的**超集**：`stamp`/`frame_id`/`source`/`count`/`detections`
   语义完全一致，几何信息放在 `geometry` / `measurement` 子字典里。**新增字段只增不改**。
5. **位姿方向约定**：`/slam/pose` 是 **Twc（世界<-相机）**，与 `/slam/path`、TF `map->camera` 一致。
   ORB-SLAM3 内部返回 `Tcw`，C++ 补丁里做了 `.inverse()`。搞反是最容易犯的错。
6. **单目尺度**：SLAM 地图单位不是米（尺度由初始化基线决定）。所以
   `measurement.method == "known_size*"` 得到的是**米**，而 `scale_fallback` 地标在**相机系**；
   只有 `triangulated` 地标与点云/轨迹同尺度。消息里 `source`/`method` 字段就是在标注这件事。

---

## 4. 接口契约速查

### 4.1 话题

| 话题 | 类型 | 谁发布 | 说明 |
|---|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | v4l2_camera / image_sequence_player | 原始图像 |
| `/image_enhanced` | `sensor_msgs/Image` | `image_enhancer_cpp` | 增强后（SLAM 与检测的推荐输入） |
| `/detections` | `std_msgs/String`(JSON) | `cube_detector` | 几何旁路检测结果（契约见 4.2） |
| `/detections/annotated` | `sensor_msgs/Image` | `cube_detector` | 标注图（RViz/rqt 看效果） |
| `/detections/markers` | `visualization_msgs/MarkerArray` | `cube_detector` | 相机系下的方块+文字 |
| `/yolo/detections` | `std_msgs/String`(JSON) | `yolo_detector`（既有） | 同契约，可替换几何旁路 |
| `/mono_py_driver/img_msg` | `sensor_msgs/Image` | `slam_input_bridge` | **ORB-SLAM3 硬编码输入** |
| `/mono_py_driver/timestep_msg` | `std_msgs/Float64` | `slam_input_bridge` | 秒，严格递增 |
| `/mono_py_driver/experiment_settings` | `std_msgs/String` | `slam_input_bridge` | 发送设置名，触发 C++ 端初始化 |
| `/mono_py_driver/exp_settings_ack` | `std_msgs/String` | ORB-SLAM3 节点 | 内容 `ACK` |
| `/slam/pose` | `geometry_msgs/PoseStamped` | ORB-SLAM3（补丁） | **Twc**，frame_id=`map` |
| `/slam/path` | `nav_msgs/Path` | ORB-SLAM3（补丁） | 累积轨迹（Twc） |
| `/slam/map_points` | `sensor_msgs/PointCloud2` | ORB-SLAM3（补丁） | x/y/z FLOAT32，默认 1 Hz |
| `/slam/status` | `std_msgs/String`(JSON) | ORB-SLAM3（补丁） | `tracking_state` 等 |
| `/semantic/landmarks` | `std_msgs/String`(JSON) | `semantic_landmark` | 地标列表（含 `source`） |
| `/semantic/markers` | `visualization_msgs/MarkerArray` | `semantic_landmark` | 绿=三角化，黄=fallback |
| `/calibration/{debug,status}` | `Image` / `String` | `calibration_node` | 标定可视化与状态 |

### 4.2 `/detections` JSON（`schema_version: 1`）

```json
{
  "schema_version": 1,
  "stamp": {"sec": 12, "nanosec": 340000000},
  "frame_id": "camera_optical",
  "source": "geometry",
  "count": 1,
  "image": {"width": 1280, "height": 480},
  "detections": [{
    "class_id": 0, "class_name": "cube", "confidence": 0.91, "track_id": 3,
    "xyxy": [586.0, 181.0, 696.0, 303.0],
    "score_terms": {"fill": 1.0, "convexity": 1.0, "aspect": 1.0, "edge": 0.62, "color": 0.95, "area": 0.93},
    "geometry": {
      "center": [640.0, 240.0], "corners": [[...]], "vertex_count": 6,
      "area_px": 8123.0, "fill_ratio": 0.87, "convexity": 0.99, "extent": 0.72,
      "aspect_ratio": 0.85, "angle_deg": 12.3, "side_px": 110.6, "long_side_px": 130.2,
      "edge_direction_count": 3, "interior_edge_count": 7, "edge_support": 0.21,
      "internal_corner_response": 0.18, "color_purity": 0.95,
      "mean_hue": 2.1, "mean_saturation": 180.0, "mean_value": 205.0
    },
    "measurement": {
      "method": "known_size_refined", "quality": 0.72, "distance_m": 0.453,
      "bearing_rad": [0.01, -0.02], "ray_camera": [0.01, -0.02, 0.999],
      "pose_rvec": [0.44, 0.55, 0.20], "pose_tvec": [0.0, 0.0, 0.45], "fit_error_px": 0.46
    }
  }]
}
```

字段含义与"哪些字段可以信"：
* `confidence` 是**几何+颜色证据的加权评分**，不是模型概率；各分量在 `score_terms` 里，调参时看它。
* `geometry.vertex_count`：4=正视单面，5/6=斜视多面。是判断"像不像立方体"的第一眼指标。
* `measurement.method`：`known_size_pose`（4 角点 PnP）| `known_size_refined`（轮廓拟合，斜视也准）
  | `known_size`（正视近似，斜视有 ~25% 系统误差）| `bearing_only`（无内参/无边长）。
  只有前三种且 `quality` 合理时，`distance_m` 才可用。

### 4.3 ROS 参数（节选，完整见 `config/*.yaml`）

`cube_detector`：`color_preset`、`hsv_ranges`(6N 个整数)、`mode`(any/square_face/cube_silhouette)、
`min_area_px`、`fill/convexity/aspect` 门限、`min_confidence`、`use_silhouette_fit`、
`cube_edge_m`、`camera_fx/fy/cx/cy`、`enable_tracking`、`max_rate_hz`。

`slam_input_bridge`：`image_topic`、`target_*`（**不要改**，C++ 硬编码）、`settings_name`、
`use_header_stamp`、`min_dt_s`、`handshake_retry_s`、`rehandshake_period_s`。

`semantic_landmark`：`min_views`、`min_parallax_deg`、`min_baseline_m`、`max_reproj_error_px`、
`stale_after_s`、`allow_scale_fallback`、`cube_edge_m`、`camera_f*`。

`ros2_orb_slam3`（补丁新增）：`publish_pose/path/points/tf/status`、`map_frame_id`、`camera_frame_id`、
`max_path_length`、`point_publish_period_ms`、`enable_pangolin`(**默认 false**)、`enable_opencv_window`。

### 4.4 C++ 补丁（唯一被修改的第三方代码）

* 改动文件：`include/ros2_orb_slam3/common.hpp`、`src/common.cpp`、`CMakeLists.txt`、`package.xml`。
* 新增话题/参数见 4.1/4.3；同时修了 3 个既有缺陷：`pAgent` 未初始化（野指针）、析构未判空、
  `enablePangolinWindow` 硬编码 true（无显示器会挂）。
* 回滚：`git -C ~/underwater_slam_project/orb_slam3_ws/src/ros2_orb_slam3 checkout -- include src CMakeLists.txt package.xml`
  或 `git apply -R patches/ros2_orb_slam3_slam_publishers.patch`。
* 细节与自检命令见该包的 `SLAM_PUBLISHERS.md`。

---

## 5. 已验证 vs 未验证（**请勿越界陈述**）

### 已在开发机真实验证（合成数据 + pytest）

* 几何识别全链路：颜色分割 → 轮廓 → 几何验证 → 打分 → NMS → 跟踪 → 测量（检出率/IoU 见表）。
* 单帧测距：包含斜视视角的轮廓拟合，误差 < 1%（合成数据）。
* 误检抑制：背景/蓝色物体 0 误检；严格模式拒绝红色平面方块。
* 标定数学与输出文件：内参恢复误差 < 1%，四件套文件可解析且数值自洽。
* 语义地标三角化：无噪声精确、1 px 噪声下 < 5 cm；视差/基线不足时正确拒绝。
* 消息契约：JSON 能被自己解析回来，且包含既有 YOLO 节点的全部字段。
* 配置一致性：YAML 键名、launch 参数声明、`%YAML:1.0` 首行、RViz 话题名等静态检查。

### 未验证（必须在 Linux 真机上做，做完请更新本节）

| 项目 | 风险 | 验证方法 |
|---|---|---|
| 所有 ROS 节点能否启动 | 中（导入、参数声明、QoS） | `ros2 launch underwater_perception full_demo.launch.py` |
| `image_bridge` 的编码处理 | 低（bgr8/mono8/rgb8 覆盖） | 用真实相机跑 `image_source:=camera`，看 `/detections/annotated` |
| launch 文件在 Humble 的兼容性 | 中（`choices=` 参数、`PythonExpression`） | 见第 6 节故障手册第 1 条 |
| C++ 补丁能否编译 | ✅ 已通过（2026-11-19，2m07s） | `colcon build`；`ros2 topic hz /slam/pose` |
| `TransformBroadcaster` 构造重载 | 中 | 编译报错就按 `SLAM_PUBLISHERS.md §9` 改成 `shared_from_this()` 并移出构造函数 |
| SLAM 输入桥接与 C++ 握手 | 中 | `ros2 topic echo /mono_py_driver/exp_settings_ack` 应看到 `ACK` |
| 真实水下图像的检出率 | **未知** | 采集 50~100 张真实图，用 `cube_detect_cli --input <dir> --print-mask` 统计 |
| 真实水下 HSV 阈值 | **未知** | 用 `--print-mask` 看掩码，按 `config/cube_detector.yaml` 头部注释微调 |
| 水下标定精度 | 未知 | 按 `docs/CALIBRATION.md` 实测，关注 RMS 与去畸变后边缘是否变直 |
| 合成画面能让 ORB-SLAM3 初始化 | **预期会失败** | 合成图纹理太少，属正常；请用真实录像验证 SLAM |

---

## 6. 故障手册（症状 → 命令 → 根因）

1. **`ros2 launch` 立刻报 `launch parameter 'X' not found` / `include launch description failed`**
   * 先看 `ros2 launch underwater_perception underwater_pipeline.launch.py -s`（只打印参数不启动）。
   * 根因通常是 `LaunchConfiguration` 没声明就被使用（开发时踩过一次）。
     `test/test_configs.py::test_launch_files_parse_and_declare_their_arguments` 就是防这个的。
2. **`/detections` 一直空，但 `/detections/annotated` 有图**
   * `ros2 topic echo /detections --once` 看 `count`。
   * 若是 0：`color_preset`/`hsv_ranges` 不匹配现场光照 → 用
     `python3 -m underwater_perception.tools.cube_detect_cli --input <单张图> --print-mask` 看掩码。
   * 若掩码正常但仍为 0：`min_area_px` 太大、`min_confidence` 太高，或 `mode=cube_silhouette`
     而目标只露一个面（改成 `any`）。
   * 若是"只在第二帧才出现"：这是 `track_min_hits=2` 的设计（抑制单帧误检），不是 bug。
3. **`/slam/pose` 没有数据 / 一直 `NOT_INITIALIZED`**
   * `ros2 topic echo /slam/status --once` 看 `tracking_state`。
   * `ros2 topic hz /mono_py_driver/img_msg`（无数据 → 桥接没收到图或没收到 ACK）。
   * `ACK` 拿不到 → 设置文件路径错：`settings_file_path_arg` 必须是**目录**，
     且 `<目录>/<settings_name>.yaml` 存在（用 `ls` 直接确认）。
   * 水下纹理少是主要原因：提高 `ORBextractor.nFeatures`、降低 `iniThFAST`，
     并确保相机**有平移**（纯旋转无法三角化初始化）。
4. **RViz 里看不到点云/轨迹**
   * Fixed Frame 必须是 `map`（slam.rviz 已设好）；`ros2 topic hz /slam/map_points` 确认有数据（默认 1 Hz）。
   * 若换了 `map_frame_id`，RViz 与 `/semantic/markers` 的 frame 都要跟着改。
5. **语义地标一直是 `scale_fallback`（黄色）**
   * 说明没有可用位姿：要么 SLAM 没初始化，要么 `/slam/pose` 的 frame_id 与检测的 frame 对不上。
   * 黄标位置在**相机系**，不要拿它当地图坐标用。
6. **标定时"采了图但标定说视图不足"**
   * 检查 `config/calibration.yaml` 的 `cols/rows` 是否与实物一致（是**内角点**数），
     `square_size` 单位是**米**（25 mm → 0.025）。
   * 用 `--dump-config` 确认参数真的传进去了。
7. **Windows 上编辑过的文件在 Linux 上出现 `\r\n` 或乱码**
   * 本仓库文件约定 UTF-8 无 BOM + LF；若用 Windows 工具改过，先
     `dos2unix <file>` 处理，再 `colcon build`。

---

## 7. 常见扩展任务（照着改）

* **换立方体颜色**：改 `config/cube_detector.yaml` 的 `color_preset`（red/orange/yellow/green/cyan/blue/magenta）
  或直接给 `hsv_ranges`（6 个一组）。非红色时把 `require_red_dominance` 设为 `false`。
* **放宽召回**：`mode: any` + 降低 `min_confidence`(0.35) + 降低 `min_area_px`。
  **收紧精度**：`mode: cube_silhouette`（要求内部棱线证据）+ 提高 `min_fill_ratio`。
* **换成 YOLO 检测器**：让 `yolo_detector` 发布到 `/detections`（参数 `detections_topic`），
  或让 `semantic_landmark` 的 `detections_topic` 指向 `/yolo/detections`。契约相同，无需改下游。
* **加一个新类别（例如圆形浮标）**：在 `core/cube_geometry.py` 旁边新增
  `core/marker_geometry.py`，产出同样的 `Detection` 对象；节点侧复用
  `cube_detector_node.py` 的结构（参数、发布、Marker）。**务必先写合成数据测试**。
* **提高测距精度**：优先保证内参准确（标定）与 `cube_edge_m` 准确；
  斜视场景确认走的是 `known_size_refined`（看 `measurement.method`）。
  要更准就靠多视图三角化（`semantic_landmark`），而不是单帧。
* **改性能**：`use_silhouette_fit: false`（斜视测距退化到 ~25% 误差，但省 ~30 ms/帧）、
  `detect_width: 960`（先缩放再检测，坐标会自动还原）、`max_rate_hz: 10`。

---

## 8. 已知陷阱清单（都是开发时真实踩过的坑，改动前请看）

| # | 陷阱 | 症状 | 现在的处理 |
|---|---|---|---|
| 1 | `cv2.solvePnP(IPPE_SQUARE)` 角点顺序不匹配 | **`ok=True` 但位姿是垃圾**（实测距离 0.005 m vs 真值 0.45 m） | `cv_compat.solve_pnp_square` 校验深度与重投影；`measurement.face_pose_from_quad` 枚举 8 种顺序取最优 |
| 2 | 轮廓拟合残差向量长度随投影顶点数变化 | 代价函数不可比，优化器**一步都不动**（表现为"拟合结果=初值"） | 轮廓按弧长重采样成固定点数（`SILHOUETTE_SAMPLES`） |
| 3 | 用"点-轮廓距离 + LM"拟合立方体 | 陷入局部极小：残差 5 px 但距离误差 10~20%，加迭代无改善 | 改为"循环对应 + PnP 迭代"（ICP 思想）+ 多起点 |
| 4 | 打磨用的合成棋盘格没有白色静区 | 角点检测锁到偏移/外扩的网格（48 个点看着"全对"） | `synthetic.render_chessboard_scene` 默认画静区；`calibration.validate_corners` 做栅格几何校验 |
| 5 | `chessboard_object_points` 坐标轴转置 | 标定仍能收敛（平面法向不可观测），但"检出 vs 真值"逐点比较差上百像素，排查方向被带偏 | 坐标轴约定写在 docstring 里；测试改为**点集比较** |
| 6 | 独立图片上开启 `min_hits=2` 跟踪 | 每张图都是新轨迹 → 首帧检测全被压掉（实测 6 张只报 3 张） | `cube_detect_cli` 默认关跟踪，`--track` 才开 |
| 7 | ROS 标定节点 `ViewResult.object_points=None` | 采集"成功"但 `calibrate()` 认为 0 个可用视图 | 采集时立刻补 `object_points`；`_add_view_from_detection` 注释说明 |
| 8 | launch 里用了未声明的 `LaunchConfiguration` | launch 执行到该 Node 时报错 | 声明集中在前；`test_configs.py` 静态校验 |
| 9 | ORB-SLAM3 设置文件注释写在 `%YAML:1.0` 之前 | OpenCV FileStorage 解析失败 | `%YAML:1.0` 必须是第一行（测试也会查） |
| 10 | Windows PowerShell 5.1 读写 UTF-8 文件 | 中文变乱码、换行被吞 | 只用 UTF-8 无 BOM + LF 写文件；不要用 PS 的 `Set-Content`/`-replace` 改源码 |
| 11 | 相机与图像播放器同时发布 `/image_raw` | 帧率抖动、时间戳乱序 | launch 用单一 `image_source` 参数做互斥 |
| 12 | 立方体朝向在单帧内不可唯一观测（无纹理对称物体） | 拟合出的 rvec 可能与真值差一个对称等价 | 只把 `distance`/位置当地标用；朝向仅供参考 |
| 13 | 平面 PnP 的物体原点是**可见面中心**，不是立方体中心 | 测距系统性偏小半个边长：5 cm 立方体实测 0.425 vs 真值 0.45（-5.6%） | `measure_geometry` 换算到立方体中心；**不能固定加 +h**——平面 PnP 有法向二义性，要按"中心 z 必然更大"选解（回归测试 `test_measurement_distance_refers_to_cube_center`） |
| 14 | `yolo_vrx_bridge/scripts/` 下脚本没有可执行位 | `colcon build` 成功但 `ros2 pkg executables yolo_vrx_bridge` 为空、`ros2 run` 报 No executable found | `chmod +x src/yolo_vrx_bridge/scripts/*` 后重新 `colcon build`（2026-11-07 已修复） |
| 15 | `colcon build --symlink-install` 报 `option --editable not recognized`；重复构建还可能报 `option --uninstall not recognized` | 构建失败，与代码无关（setuptools 81 移除旧命令） | 用普通 `colcon build`；重复构建前 `rm -rf build/<pkg> install/<pkg>`；或固定兼容的 setuptools 版本 |
| 16 | `enhancer_underwater.yaml` 里写 `camera_matrix: []` | C++ 节点启动即崩：`InvalidParameterValueException: No parameter value set`（ROS2 把空数组当未设置） | 标定前**注释掉**这两个键；标定后填 9/5 个实际数值（2026-11-07 已修复） |
| 17 | 时间戳走独立的 `/mono_py_driver/timestep_msg`，图像走另一个话题 | ORB-SLAM3 偶发 `Frame with a timestamp older than previous frame detected!`，然后 `NO_IMAGES_YET` / 地图反复重建；看起来像“初始化失败” | 桥接把**严格递增的时间戳写进图像 header**，C++ `TrackMonocular` 用 header 时间；timestep 话题保留仅为兼容（2026-11-19 已修复） |

---

## 9. 变更清单（本次新增/修改）

**新增**（全部在 `underwater_slam_ws/src/underwater_perception/`）：见第 1 节目录树。
**修改**：`orb_slam3_ws/src/ros2_orb_slam3/` 的 4 个文件（C++ 补丁，见 4.4）。
**未改动**：`image_enhancer_cpp`、`yolo_vrx_bridge`、`config/*`、`models/*`、原始 Readme。

验证命令汇总（复制即用）：

```bash
cd ~/underwater_slam_project/src/underwater_perception
PYTHONPATH=. python3 -m pytest test -q                      # 99 项
PYTHONPATH=. python3 -m underwater_perception.tools.selftest --out /tmp/selftest
PYTHONPATH=. python3 -m underwater_perception.tools.cube_detect_cli --synthetic --out /tmp/demo
PYTHONPATH=. python3 -m underwater_perception.tools.calibrate_cli --images <目录> --out /tmp/calib
cd ~/underwater_slam_project && colcon build --symlink-install
source install/setup.bash && ros2 launch underwater_perception full_demo.launch.py
```
