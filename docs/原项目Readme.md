# 基于单目相机与 SLAM 的水下建图项目

本项目面向“大一年度项目：基于单目相机与 SLAM 的水下建图”。当前仓库保留源码、配置、报告和已经导出的 YOLOv8 ONNX 权重。项目当前主线是：

```text
真实单目相机 / VRX-Gazebo 相机
        -> /image_raw
        -> image_enhancer_cpp 水下图像增强
        -> /image_enhanced
        -> yolo_vrx_bridge 加载 models/best.onnx 做 YOLO 检测
        -> /yolo/annotated + /yolo/detections
        -> ORB-SLAM3 / RViz / 后续语义地图显示
```

中期目标：完成相机采集、图像增强、ONNX 目标检测、标定流程和演示材料整理。  
结题目标：在真实水下环境中实现单目相机采集、抗干扰目标识别、单目 SLAM 建图，并在上位机/RViz 中返回轨迹、点云和语义目标。

## 文件和目录说明

```text
/home/hu/underwater_slam_project
├── Readme.md
├── 下一步工作Readme.md
├── 实际操作步骤.md
├── 文件目录架构.md
├── config
│   ├── enhancer_underwater.yaml
│   ├── water_1280x480.orbslam3.example.yaml
│   └── yolo_vrx_demo.yaml
├── models
│   ├── README.md
│   └── best.onnx
├── src
│   ├── image_enhancer_cpp
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   └── src/enhancer.cpp
│   └── yolo_vrx_bridge
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── launch/yolo_vrx_pipeline.launch.py
│       ├── scripts/dataset_player
│       ├── scripts/yolo_detector
│       └── yolo_vrx_bridge
│           ├── dataset_player_node.py
│           ├── image_utils.py
│           └── yolo_detector_node.py
```
说明：

- `src/image_enhancer_cpp/src/enhancer.cpp`：C++ ROS2 水下图像增强节点，订阅 `/image_raw`，发布 `/image_enhanced`，可选发布 `/image_enhanced_debug`。
- `src/yolo_vrx_bridge/yolo_vrx_bridge/yolo_detector_node.py`：YOLO 检测节点，当前可直接加载 `models/best.onnx`，不再强制依赖数据集 `data.yaml`。
- `src/yolo_vrx_bridge/yolo_vrx_bridge/dataset_player_node.py`：可选数据集回放节点。只有恢复 YOLO 格式数据集后才使用。
- `config/enhancer_underwater.yaml`：图像增强参数模板，包含红通道补偿、Gray World 白平衡、CLAHE、去畸变等参数。
- `config/water_1280x480.orbslam3.example.yaml`：ORB-SLAM3 单目配置模板，水下标定完成后把 `ost.yaml` 的内参和畸变参数写入这里。
- `models/best.onnx`：当前保留的 YOLOv8 ONNX 权重，是交付演示中检测功能的核心文件。
- `tools/generate_ppt_materials.py`：PPT 图片生成脚本。历史图片已生成在 `/home/hu/ppt_materials`；如果没有原训练数据，数据集样例图不建议重新生成。

## 环境要求

推荐环境：

- Ubuntu 22.04
- ROS2 Humble
- OpenCV
- Python3
- Ultralytics + ONNX Runtime
- 可选：VRX/Gazebo、ORB-SLAM3、RViz2

安装基础依赖：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-image-tools \
  ros-humble-v4l2-camera \
  v4l-utils \
  python3-opencv \
  python3-pil \
  python3-yaml
```

安装 YOLO 推理依赖：

```bash
pip install ultralytics onnxruntime
```

## 编译工作空间

```bash
cd /home/hu/underwater_slam_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 真实相机启动步骤

### 1. 确认相机设备

```bash
ls /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### 2. 终端 1：发布相机图像

先确认设备（**`video1`/`video3` 通常是 metadata 节点，不能取流**）：

```bash
v4l2-ctl --list-devices
# HD Webcam:      /dev/video0（取流）  /dev/video1（metadata）
# Lightburn Cam:  /dev/video2（取流）  /dev/video3（metadata）
```

**方案 A（推荐，本机实测 640x480 约 19 fps）**：用本项目的 OpenCV 相机节点，
默认 MJPG，会打印实际协商到的格式/帧率：

```bash
source /home/hu/underwater_slam_project/install/setup.bash
ros2 launch underwater_perception camera.launch.py \
  camera_index:=0 width:=1280 height:=720 fourcc:=MJPG fps:=30.0
```

> 1280x720 的 MJPG 在 Python/DDS 下大约 7~8 fps；需要更高帧率时用
> `width:=640 height:=480`（约 19 fps），或只在标定/采数据集时用不经过 ROS 的
> `capture_dataset` / `calibrate_gui`（OpenCV 直接读 1280x720 MJPG 约 22 fps）。

**方案 B（ROS 原生驱动）**：本机 `v4l2_camera 0.6.2` **不支持 MJPG**，
1280x720 只能用 YUYV（约 10 fps）；640x480 约 10 fps。可用但不推荐做高帧率实验：

```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=/dev/video0 \
  -p image_size:="[640, 480]" \
  -p pixel_format:=YUYV \
  -r image_raw:=/image_raw
```

**为什么 1280x720 这么低**：YUYV 是未压缩格式，USB2（480 Mbps）带宽下
1280x720 只能到 10 fps（Lightburn 相机甚至只有 5 fps）；MJPG 压缩后才有 30 fps 的协商能力。
主机和相机当前都在 USB2 总线上（`lsusb -t` 显示 480M），换 USB3 口或换 MJPG 才能改善。

### 3. 终端 2：启动水下图像增强

```bash
cd /home/hu/underwater_slam_project
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run image_enhancer_cpp enhancer --ros-args \
  --params-file /home/hu/underwater_slam_project/config/enhancer_underwater.yaml
```

也可以直接传参：

```bash
ros2 run image_enhancer_cpp enhancer --ros-args \
  -p input_topic:=/image_raw \
  -p output_topic:=/image_enhanced \
  -p publish_debug_view:=true \
  -p use_red_compensation:=true \
  -p use_gray_world:=true \
  -p use_clahe:=true \
  -p clahe_clip_limit:=3.0
```

### 4. 终端 3：启动 ONNX 检测

```bash
cd /home/hu/underwater_slam_project
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run yolo_vrx_bridge yolo_detector --ros-args \
  -p input_topic:=/image_enhanced \
  -p model_path:=/home/hu/underwater_slam_project/models/best.onnx \
  -p annotated_topic:=/yolo/annotated \
  -p detections_topic:=/yolo/detections \
  -p confidence_threshold:=0.25
```

### 5. 查看图像与检测结果

```bash
ros2 topic list
ros2 topic echo /yolo/detections --once
ros2 run rqt_image_view rqt_image_view
```

推荐查看：

- `/image_raw`：相机原图。
- `/image_enhanced`：水下增强图像。
- `/image_enhanced_debug`：原图和增强图拼接对比。
- `/yolo/annotated`：ONNX 检测结果图。
- `/yolo/detections`：JSON 格式检测框、类别和置信度。

## 水下标定步骤

水下标定命令：

```bash
ros2 run camera_calibration cameracalibrator \
  --square 0.025 \
  --size 8x6 \
  --ros-args -p image:=/image_enhanced camera:=/camera
```

标定完成后，把 `ost.yaml` 中的参数写入 ORB-SLAM3 配置：

```yaml
Camera.fx: xxx
Camera.fy: xxx
Camera.cx: xxx
Camera.cy: xxx
Camera.k1: xxx
Camera.k2: xxx
Camera.p1: xxx
Camera.p2: xxx
Camera.k3: xxx
```

注意：平面防水壳会引入空气-玻璃-水多介质折射，普通针孔模型只是近似。当前阶段先用水下标定结果提高可用性，后续可考虑折射模型或圆顶端口。

## VRX/Gazebo 接入

本机已有 VRX 资源：

```text
/home/hu/vrx_humble.zip
/home/hu/vrx-humble
```

如需运行 VRX：

```bash
cd /home/hu/vrx-humble
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch vrx_gz competition.launch.py world:=perception_task sim_mode:=full headless:=False
```

VRX 相机话题常见形式：

```text
/wamv/sensors/cameras/front_left_camera_sensor/image_raw
```

接入增强节点：

```bash
ros2 run image_enhancer_cpp enhancer --ros-args \
  -p input_topic:=/wamv/sensors/cameras/front_left_camera_sensor/image_raw \
  -p output_topic:=/image_enhanced
```

再启动 YOLO 检测节点即可复用 `models/best.onnx`。

## 关于数据集和复训

当前交付版本只保留导出的 `models/best.onnx`。因此：

导出后可替换或新增到 `models/`，运行检测时把 `model_path` 指向新文件。

## 如何提高水下检测能力

1. 图像输入优化：使用红通道补偿减轻偏蓝偏绿，使用 Gray World 白平衡稳定颜色，使用 CLAHE 提升局部对比度，并根据水体浑浊程度调整 `clahe_clip_limit`。
2. 标定优化：在水下重新标定相机，把内参和畸变参数同步到增强节点与 ORB-SLAM3，减少检测框中心、特征点和几何位姿之间的不一致。
3. 数据优化：补充真实水池/水箱图像、不同光照与浊度图像、负样本、小目标和遮挡样本，提高模型对水下环境变化的适应能力。
4. 模型优化：从 YOLOv8n 或 YOLOv8s 开始，先保证实时性，再比较不同输入尺寸、置信度阈值和增强策略。
5. SLAM 联动：把动态目标检测框转换为 mask，在 ORB 特征提取时剔除不稳定区域；对静态目标进行多帧三角测量，作为语义地标显示到 RViz。

## 后续主线

1. 用防水棋盘格完成真实水下标定，更新 `ost.yaml` 与 ORB-SLAM3 配置。
2. 使用真实相机跑通 `/image_raw -> /image_enhanced -> /yolo/annotated`。
3. 使用 `models/best.onnx` 做水下目标检测初测，并记录置信度、误检、漏检和 FPS。
4. 接入 ORB-SLAM3 单目，先实现轨迹和稀疏点云。
5. 将 YOLO 检测框转换为动态 mask，测试是否能减少 SLAM 跟踪丢失。
6. 在 RViz 中显示增强图像、检测框、轨迹、点云和语义目标。
7. 补采或恢复训练数据，重新训练并导出更适合真实水下场景的 ONNX 模型。

---

## 新增：underwater_perception（几何旁路 + 标定 + SLAM 桥接 + 语义地标）

在 `src/underwater_perception/` 下新增了一个包，解决三件事：

1. **几何旁路识别**（YOLO 数据集未就绪时的可用方案）：用"颜色分割 + 轮廓几何 + 内部棱线结构"
   检测红色小正方体，输出与既有 YOLO 节点**兼容的统一 JSON 契约**（`/detections`），
   数据集好了以后直接换检测器即可，下游不用改。
2. **参数化标定程序**：棋盘格 8x6 / 25 mm 为默认值，参数全部可改；
   支持图片文件夹 / 视频 / USB 相机 / ROS 话题四种来源；
   一次性输出 `ost.yaml`、ORB-SLAM3 设置文件、增强节点内参、JSON 报告。
3. **架构补全**：活体相机 → 增强 → 识别 → SLAM 的桥接节点（原仓库自带的驱动只能回放
   EuRoC 数据集）、ORB-SLAM3 位姿/轨迹/点云发布补丁、多视图三角化的语义地标、RViz 配置、
   一键 launch（含**不需要相机**的合成验证链路）。

**先看这两份文档**：

* `src/underwater_perception/docs/AI_HANDOFF.md` —— 交接总纲（含已实测数据、故障手册、已知陷阱）
* `src/underwater_perception/docs/UNVERIFIED.md` —— **哪些还没验证**（在 Linux + 真机上需要补做）

**最快的验证方式（无需相机、无需数据集）**：

```bash
cd ~/underwater_slam_project
PYTHONPATH=src/underwater_perception python3 -m underwater_perception.tools.selftest --out ./selftest_out
cd src/underwater_perception && PYTHONPATH=. python3 -m pytest test -q && cd ../..
colcon build --symlink-install && source install/setup.bash
ros2 launch underwater_perception full_demo.launch.py
```

**标定（水下标定，标定板与相机都在水下）**：

```bash
python3 -m underwater_perception.tools.calibrate_cli \
  --images ~/calib_images --cols 8 --rows 6 --square 0.025 \
  --out ~/underwater_slam_project/calibration_out
```

**SLAM 建图链路**：

```bash
ros2 launch underwater_perception slam_pipeline.launch.py \
  voc_file:=$HOME/orb_slam3_ws/src/ros2_orb_slam3/orb_slam3/Vocabulary/ORBvoc.txt.bin \
  settings_dir:=$HOME/orb_slam3_ws/src/ros2_orb_slam3/orb_slam3/config/Monocular/ \
  settings_name:=water_1280x480 rviz:=true
```

详细步骤、参数含义与故障排查见 `docs/RUNBOOK.md`、`docs/CALIBRATION.md`、`docs/INTERFACES.md`。


---

## 新增：真实环境工具链（2026-11）

| 功能 | 入口 | 说明 |
|---|---|---|
| 相机取流（MJPG） | `ros2 launch underwater_perception camera.launch.py` | 替代 v4l2_camera 的慢速 YUYV 转换；会打印实际格式/帧率 |
| 图形化标定（标定板参数可改） | `ros2 run underwater_perception calibrate_gui` | 图案/内角点/方格毫米数可改；USB 相机/视频/图片三种来源 |
| ROS 数据集采集 | `ros2 launch underwater_perception dataset_capture.launch.py` | 按间隔把 `/image_enhanced` 帧存成 YOLO 数据集 |
| 独立数据集采集 | `python3 -m underwater_perception.tools.capture_dataset --camera 0 --fourcc MJPG --fps 30` | 无 ROS 时直接读 USB 相机/视频；video1/video3 常是 metadata |
| 数据集体检 | `python3 -m underwater_perception.tools.yolo_dataset_check --dataset <目录>` | 缺失标签/非法行/跨集重复检查 |
| 真实环境 YOLO | `ros2 launch underwater_perception yolo_real.launch.py model_path:=.../best.pt` | 当前 `models/best.pt` 已由 rubbish 数据集训练 |
| 训练脚本 | `python3 scripts/train_yolo.py --data <数据集>` | 自动切 valid、训练并复制 `models/best.pt` |
| ONNX 导出脚本 | `python3 scripts/export_yolo_onnx.py --weights models/best.pt` | 备份旧模型、导出并自检 `models/best.onnx` |
| 算法审查报告 | `src/underwater_perception/docs/思路报告.md` | 问题清单、改进路线、PPT 素材清单 |
| YOLO 真实数据全流程 | `src/underwater_perception/docs/YOLO_REAL_DATA.md` | 采什么、怎么标、怎么拆、怎么训、怎么接 SLAM |

> 注意：仓库**没有**提供可用的 `best.pt`。训练完成前检测节点会显示
> `MODEL NOT LOADED`，不会用标签回放冒充真实检测。

---

## 一键真实水下建图（当前主线，2026-11-19）

> VRX/Gazebo 不再作为项目主线，以下命令全部使用真实相机。

```bash
# 首次使用先编译 ORB-SLAM3（已编译可跳过）
cd ~/underwater_slam_project
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_orb_slam3

source ~/underwater_slam_project/install/setup.bash
source ~/underwater_slam_project/install/setup.bash

# 相机 -> 增强 -> YOLO(best.pt) -> 桥接 -> ORB-SLAM3 -> 语义地标 -> RViz
ros2 launch underwater_perception mapping.launch.py

# 无显示器 / 远程
ros2 launch underwater_perception mapping.launch.py rviz:=false

# 同时打开 OpenCV 2D 俯视图（轨迹 + 地图点 + 语义地标）
ros2 launch underwater_perception mapping.launch.py viewer:=true
```

单独运行 2D 可视化（右下角带 `/image_enhanced` 相机预览）：

```bash
ros2 run underwater_perception slam_viewer
```

录制一段建图数据（用于回放、截图、结题材料）：

```bash
bash scripts/record_mapping.sh
# 结束后回放
ros2 bag play ~/underwater_slam_project/bags/mapping_<时间戳>
ros2 run underwater_perception slam_viewer
```

采集/训练模型：

```bash
python3 scripts/train_yolo.py --data /home/hu/Downloads/rubbish.v1i.yolov8 --epochs 100 --copy-last
python3 scripts/export_yolo_onnx.py --weights ~/underwater_slam_project/models/best.pt
```

关键检查：

```bash
ros2 topic echo /slam/status --once
ros2 topic hz /slam/pose
ros2 topic hz /slam/map_points
ros2 topic echo /yolo/detections --once
```

若 `tracking_state` 一直是 `NOT_INITIALIZED`：确认画面有纹理，并且相机在**平移**而不是原地旋转。
换用 `/dev/video2` 时，必须用该相机自己的标定文件，不能直接用 `water_1280x480.yaml`（那是 video0 的内参）。
