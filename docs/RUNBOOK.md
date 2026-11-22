# 操作手册（RUNBOOK）

所有命令假设：

```bash
source /opt/ros/humble/setup.bash
source ~/underwater_slam_project/install/setup.bash
source ~/underwater_slam_project/install/setup.bash
```

> 建议把这三行写进 `~/.bashrc`（或每次开终端执行一次），否则 `ros2 run` 找不到包。

> **2026-11-19 起主线已切换为真实相机；VRX/Gazebo 相关内容只保留为历史资料，不再使用。**

---

## 场景 0：真实相机一键建图（当前主线）

```bash
source /opt/ros/humble/setup.bash
source ~/underwater_slam_project/install/setup.bash
source ~/underwater_slam_project/install/setup.bash

# 相机 -> 增强 -> YOLO -> 桥接 -> ORB-SLAM3 -> 语义地标 -> RViz
ros2 launch underwater_perception mapping.launch.py

# 远程/无显示器
ros2 launch underwater_perception mapping.launch.py rviz:=false

# 额外打开 OpenCV 2D 俯视图（轨迹 + 地图点 + 语义地标）
ros2 launch underwater_perception mapping.launch.py viewer:=true

# 没有可用 best.pt 时先不跑 YOLO
ros2 launch underwater_perception mapping.launch.py use_yolo:=false
```

**必须在画面有纹理时缓慢平移相机**，单目纯旋转无法初始化。
看到 `/slam/status` 从 `NOT_INITIALIZED` 变成 `OK` 后，RViz 的轨迹/点云开始增长。

> **UVC 编号会变**：Lightburn 相机重新插拔/反复打开后，可能从 `/dev/video2` 变成
> `/dev/video3`。用稳定路径避免踩坑：
> ```bash
> ls -l /dev/v4l/by-id/*Lightburn*
> ros2 launch underwater_perception mapping.launch.py \
>   camera_device:=/dev/v4l/by-id/usb-WN_210726_Lightburn_Camera_01.00.00-video-index0 \
>   use_yolo:=false use_landmark:=false rviz:=false
> ```
> `camera_device` 优先级高于 `camera_index`；默认值没有改。
>
> **相机 2（Lightburn / `/dev/video2`）不能复用相机 0 的标定文件。**
> `config/orbslam3/water_1280x480.yaml` 的 fx/fy 来自 HD Webcam；
> 换相机后必须用 `calibrate_gui` 对 `/dev/video2` 重新标定，
> 而且标定分辨率要与启动分辨率一致（1280x720 或 640x480）。
> 否则会出现“有图但一直 NOT_INITIALIZED / 跟踪立刻丢失”。

> **同一时间只能跑一套 `mapping.launch.py`。** 同时启动两套会抢同一个 UVC 相机，
> 内核会把 Lightburn 从 `/dev/video2` 重枚举到 `/dev/video3`，旧句柄失效后表现为
> “读帧失败/整个链路卡住”。相机节点现在会在连续读失败时自动重连，但仍应先停掉旧的一套。

> **`slam_viewer` 退出请用 `Ctrl+C`，不要 `Ctrl+Z`。** `Ctrl+Z` 只是挂起进程（状态 T），
> 窗口会卡住不动；恢复用 `fg`，结束用 `kill`。

**性能注意（1280x720 实测会打满 CPU）**：
`camera` 与 `slam_input_bridge` 在 1280x720 下各占约 100% CPU，RViz 还会再占约 120%；
三者叠加会把帧率拖到 1~3 fps，ORB-SLAM3 无法初始化。建议：

```bash
# 建图时先不开 RViz
ros2 launch underwater_perception mapping.launch.py \
  camera_index:=2 camera_width:=1280 camera_height:=720 \
  use_yolo:=false use_landmark:=false rviz:=false \
  point_publish_period_ms:=2000

# 需要看画面时用轻量窗口
ros2 run underwater_perception slam_viewer
```

如果仍低于约 5 fps，改用 640x480（必须先对 video2 做 640x480 标定），
数据量降到 1/3，通常能到 15~20 fps：

```bash
# 用 calibrate_gui 对 camera 2 / 640x480 标定，得到 water_cam2_640.yaml
cp calibration_cam2_640/water_cam2_640.yaml config/orbslam3/

ros2 launch underwater_perception mapping.launch.py \
  camera_index:=2 camera_width:=640 camera_height:=480 \
  settings_name:=water_cam2_640 \
  use_yolo:=false use_landmark:=false rviz:=false
```

只开可视化（不依赖 RViz）：

```bash
ros2 run underwater_perception slam_viewer
```

`slam_viewer` 窗口右下角会显示 `/image_enhanced` 相机画面；RViz 的 `slam.rviz`
默认也显示 `/image_enhanced`，YOLO 标注图默认关闭（纯几何建图不启动 YOLO）。

---

## 场景 A：没有相机，先确认代码是对的（**第一次上手请做这个**）

```bash
cd ~/underwater_slam_project

# A1 纯算法自测（不需要 ROS）
PYTHONPATH=src/underwater_perception python3 -m underwater_perception.tools.selftest --out ./selftest_out
# 期望：6/6 项 PASS，报告见 selftest_out/selftest_report.json

# A2 单元测试
cd src/underwater_perception && PYTHONPATH=. python3 -m pytest test -q && cd ../..

# A3 ROS 链路（合成场景驱动，不需要相机）
colcon build --symlink-install && source install/setup.bash
ros2 launch underwater_perception full_demo.launch.py
# 另开终端逐项检查：
ros2 topic hz /image_raw                 # ~10 Hz
ros2 topic hz /image_enhanced            # 同频
ros2 topic echo /detections --once       # count=1，confidence>0.5
ros2 topic hz /mono_py_driver/img_msg    # 桥接在工作
```

**预期**：A1/A2 全绿；A3 的前 4 项有数据。第 5 项（SLAM）在合成画面上**大概率停在
`NOT_INITIALIZED`**，这是正常的（合成图纹理太少），不要误判为代码错误。

### A4：无相机验证完整 SLAM（推荐）

用 ORB-SLAM3 自带的 EuRoC 运动序列驱动完整链路，能真正看到轨迹和点云：

```bash
ros2 launch underwater_perception slam_demo.launch.py
# 另开终端
ros2 topic echo /slam/status --once        # 应为 OK
ros2 topic hz /slam/path                   # ~10 Hz
ros2 topic hz /slam/map_points             # ~1 Hz
ros2 run underwater_perception slam_viewer # 2D 俯视图
```

实测：`tracking_state=OK`、4464 地图点、pose 9.9 Hz、path 10 Hz、points 0.93 Hz。

---

## 场景 B：真实相机的最小可运行链路

```bash
# B1 确认相机与格式
ls /dev/video*
v4l2-ctl -d /dev/video0 --list-formats-ext

# B2 启动：相机 -> 增强 -> 识别
ros2 launch underwater_perception underwater_pipeline.launch.py \
  image_source:=camera video_device:=/dev/video0 rviz:=true

# B3 看效果（任选）
ros2 run rqt_image_view rqt_image_view        # 选 /detections/annotated
ros2 topic echo /detections --once
```

调参（现场常用）：

```bash
# 只看掩码，判断颜色阈值是否合适
python3 -m underwater_perception.tools.cube_detect_cli --input <随手截的一张图> --print-mask
# 建议把该图的输出（annotated_/mask_overlay_）打开对照，然后改 config/cube_detector.yaml
```

---

## 场景 C：水下标定（**先标定再跑 SLAM**）

```bash
# C1 采集：20~30 个角度，标定板与相机都在水下，水质接近实验条件
mkdir -p ~/calib_images
# 用相机逐张抓图（示例：用 v4l2 抓 30 张，或直接用标定节点的自动采集）
ros2 launch underwater_perception calibration.launch.py image_topic:=/image_enhanced
ros2 run rqt_image_view rqt_image_view         # 看 /calibration/debug，角点应稳定
ros2 service call /calibration_node/run std_srvs/srv/Trigger "{}"   # 够了就标定

# C2 或者离线标定（推荐：图片可控、可重标）
python3 -m underwater_perception.tools.calibrate_cli \
  --images ~/calib_images --cols 8 --rows 6 --square 0.025 \
  --out ~/underwater_slam_project/calibration_out
# 关注输出：RMS、有效视图数、误差最大的视图、警告

# C3 把结果装进链路
cp ~/underwater_slam_project/calibration_out/water_1280x480.yaml \
   ~/underwater_slam_project/src/ros2_orb_slam3/orb_slam3/config/Monocular/
# 把 calibration_out/enhancer_camera_params.yaml 的 camera_matrix / distortion_coeffs
# 填进 config/enhancer_underwater.yaml 并把 use_undistort 改成 true
```

细节与判读见 `CALIBRATION.md`。

---

## 场景 D：SLAM 建图 + 语义地标

```bash
# D1 启动 SLAM 链路（注意两个路径参数）
ros2 launch underwater_perception slam_pipeline.launch.py \
  voc_file:=$HOME/orb_slam3_ws/src/ros2_orb_slam3/orb_slam3/Vocabulary/ORBvoc.txt.bin \
  settings_dir:=$HOME/orb_slam3_ws/src/ros2_orb_slam3/orb_slam3/config/Monocular/ \
  settings_name:=water_1280x480 rviz:=true

# 另开终端再启动感知（相机 -> 增强 -> 识别 -> 桥接）
ros2 launch underwater_perception underwater_pipeline.launch.py image_source:=camera

# D2 逐项确认
ros2 topic echo /mono_py_driver/exp_settings_ack --once     # ACK
ros2 topic echo /slam/status --once                          # tracking_state: OK
ros2 topic hz /slam/pose                                     # 与图像同频
ros2 topic hz /slam/map_points                                # 默认 1 Hz
ros2 topic echo /semantic/landmarks --once                    # source: triangulated（有位姿后）
```

**操作要点（水下单目初始化）**
* 初始阶段要有**平移**，并且画面里有足够纹理（池底、墙面、石块）；
  纯旋转或纯色背景会让单目 SLAM 无法初始化。
* 移动要**慢**：水下能见度低、特征少，快速运动会直接丢失跟踪。
* 若一直 `NOT_INITIALIZED`：提高 `ORBextractor.nFeatures`（1800~2500）、
  降低 `iniThFAST`（10~12），并确认图像已做增强（用 `/image_enhanced` 而不是 `/image_raw`）。

---

## 场景 E：用真实录像做可重复验证（推荐用于答辩材料）

```bash
# E1 录一段 rosbag（含 /image_enhanced 与 /detections，方便离线复盘）
ros2 bag record -o ~/bags/pool_run1 /image_raw /image_enhanced /detections \
  /slam/pose /slam/path /slam/status

# E2 回放（起一个回放 + 完整链路）
ros2 bag play ~/bags/pool_run1
# 或者把 bag 里的图像导出成图片，用 CLI 离线统计检出率：
python3 -m underwater_perception.tools.cube_detect_cli --input ~/frames --out ~/eval_out
# 关注 detections.json 里的 summary.mean_iou_vs_synthetic_truth 不适用（真实数据无真值），
# 但 detected_frames / mean_confidence / track_ids 是可汇报的指标
```

答辩/结题建议记录的量化指标（写入报告，比截图有说服力）：
帧率、增强前后特征点数、检出率与误检数、单帧测距误差（与卷尺实测对比）、
SLAM 跟踪丢失次数、回到起点时的漂移、开启/关闭增强的对比。

---

## 场景 F：紧急回滚

```bash
# F1 只回滚 C++ 补丁（保留本项目的 Python 部分）
cd ~/underwater_slam_project/src/ros2_orb_slam3
git apply -R patches/ros2_orb_slam3_slam_publishers.patch    # 或 git checkout -- <那 4 个文件>
cd ~/underwater_slam_project && colcon build --symlink-install

# F2 关掉某个新增节点：直接用 launch 参数控制
ros2 launch underwater_perception slam_pipeline.launch.py start_slam:=false   # 只跑桥接与地标
# 或者手动单起节点（不用 launch，便于看单个节点的日志）
ros2 run underwater_perception cube_detector --ros-args --params-file <绝对路径>/cube_detector.yaml

# F3 参数兜底：任何参数都可以命令行覆盖，不必改文件
ros2 run underwater_perception cube_detector --ros-args -p min_confidence:=0.35 -p mode:=any
```
