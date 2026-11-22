# 基于单目 SLAM 的水下建图（最终提交版）

本项目把原来的两个工作区合并为一个可直接构建、运行和提交的目录：

```text
~/underwater_slam_project
```

原工作区：

```text
~/underwater_slam_ws       # 相机、增强、检测、桥接、可视化、标定
~/orb_slam3_ws             # ORB-SLAM3 本体
```

已经合并到本目录的 `src/` 下，**不需要再单独 source `orb_slam3_ws`**。

---

## 1. 目录结构

```text
underwater_slam_project/
├── README.md                     # 本文件
├── src/
│   ├── underwater_perception/    # ★ 项目核心（标定/桥接/地标/相机/可视化）
│   ├── image_enhancer_cpp/       # 水下图像增强（去畸变/红补偿/白平衡/CLAHE）
│   ├── yolo_vrx_bridge/          # YOLO 检测节点（包名为历史遗留）
│   └── ros2_orb_slam3/           # ORB-SLAM3 C++ 本体 + ROS2 封装
├── config/
│   ├── enhancer_underwater.yaml  # 增强参数（权威文件之一）
│   ├── yolo_real_camera.yaml     # 真实相机 YOLO 配置
│   └── orbslam3/
│       ├── water_1280x480.yaml   # 当前 Lightburn 相机标定（1280x720）
│       └── EuRoC.yaml            # 无相机 demo 用
├── calibration/
│   ├── camera2_1280x720/         # 本次标定四件套 + undistorted 示例
│   └── old_1280x720/             # 早期标定，仅留档
├── models/                       # best.pt / best.onnx / 元数据
├── datasets/                     # 原始 13 张 + 已切分 10/3
├── runs/yolo/rubbish_v2/         # 训练曲线、混淆矩阵、验证图
├── scripts/                      # 训练、ONNX 导出、录制
├── docs/                         # 全部文档（含思路报告）
└── reports/                      # 自测报告等
```

构建产物 `build/`、`install/`、`log/`、缓存、旧权重备份、208MB EuRoC 完整测试集没有放入提交包。  
仓库自带 80 张 EuRoC 样张用于无相机验证：`src/underwater_perception/data/euroc_sample/`。

---

## 2. 环境依赖

- Ubuntu 22.04 + ROS2 Humble
- OpenCV 4.x、Eigen3、Boost、libcrypto
- Pangolin（ORB-SLAM3 需要）
- Python3：`numpy`、`opencv-python`/`python3-opencv`、`pyyaml`
- 可选训练：`ultralytics`、`torch`、`onnxruntime`、`matplotlib>=3.8`

检查 ROS：

```bash
source /opt/ros/humble/setup.bash
ros2 --help
```

---

## 3. 构建

```bash
cd ~/underwater_slam_project
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

> 第一次构建 ORB-SLAM3 约 2 分钟。
> 如果重复构建报 `option --uninstall not recognized`，删除对应
> `build/<包名>` 和 `install/<包名>` 后重试（setuptools 版本兼容问题）。

---

## 4. 真实相机一键建图

先找相机的稳定设备路径：

```bash
ls -l /dev/v4l/by-id/
```

例如 Lightburn 当前可能是：

```text
/dev/v4l/by-id/usb-WN_210726_Lightburn_Camera_01.00.00-video-index0
```

不要用 `/dev/video2`、`/dev/video3` 这种会变的编号。

```bash
cd ~/underwater_slam_project
source /opt/ros/humble/setup.bash
source install/setup.bash

CAM=$(ls /dev/v4l/by-id/*Lightburn* | head -1)

ros2 launch underwater_perception mapping.launch.py \
  camera_device:=$CAM \
  camera_width:=1280 camera_height:=720 \
  use_yolo:=false use_landmark:=false \
  rviz:=true \
  point_publish_period_ms:=2000
```

当前启动链路：

```text
相机 -> image_enhancer_cpp -> slam_input_bridge -> ORB-SLAM3 -> RViz
```

纯几何建图时：

- `use_yolo:=false`
- `use_landmark:=false`
- 不启动 YOLO，不启动语义地标。

### 可视化

RViz 会自动显示轨迹、稀疏点云、相机图。  
也可以另开终端用轻量 2D 俯视图：

```bash
source /opt/ros/humble/setup.bash
source ~/underwater_slam_project/install/setup.bash
ros2 run underwater_perception slam_viewer
```

显示：

- 绿色轨迹（X-Z 俯视）
- 灰色稀疏地图点
- 右下角 `/image_enhanced` 相机画面
- tracking 状态 / FPS / 地图点数

### 检查状态

```bash
source /opt/ros/humble/setup.bash
source ~/underwater_slam_project/install/setup.bash

while true; do
  ros2 topic echo /slam/status --once
  sleep 1
done
```

看到：

```json
{"tracking_state":"OK","tracking_state_code":2,...}
```

说明初始化成功。

**单目初始化必须有平移**，不能只原地旋转；移动要慢，画面要有纹理。

---

## 5. 无相机验证（可选）

自带 80 张 EuRoC 样张，可以直接验证整条 SLAM 链路：

```bash
source /opt/ros/humble/setup.bash
source ~/underwater_slam_project/install/setup.bash

ros2 launch underwater_perception slam_demo.launch.py
```

正常会进入 `tracking_state=OK`，并可看到轨迹/点云。

也可以用自己的图片目录：

```bash
ros2 launch underwater_perception slam_demo.launch.py \
  image_dir:=/path/to/images fps:=10.0
```

---

## 6. 标定

当前标定：

```text
calibration/camera2_1280x720/
  1280x720，25/25 视图，RMS=0.4026 px
```

已安装到：

```text
config/orbslam3/water_1280x480.yaml
src/underwater_perception/config/orbslam3/water_1280x480.yaml
```

增强节点已写入同一组内参并开启 `use_undistort: true`。

重新标定：

```bash
source /opt/ros/humble/setup.bash
source ~/underwater_slam_project/install/setup.bash
ros2 run underwater_perception calibrate_gui
```

标定完成后：

1. 把生成的 `water_*.yaml` 放到 `config/orbslam3/`；
2. 同时放到 `src/underwater_perception/config/orbslam3/`；
3. 把 `enhancer_camera_params.yaml` 的 9+5 个数填进
   `config/enhancer_underwater.yaml` 和
   `src/underwater_perception/config/enhancer_underwater.yaml`；
4. 重新 `colcon build`，让安装目录里的配置更新。

---

## 7. YOLO 数据集与模型（可选）

当前模型：

```text
models/best.pt     # 单类 rubbish
models/best.onnx   # 输入 1x3x640x640，输出 1x5x8400
```

注意：训练集只有 13 张（10 train / 3 valid），**只能证明流程可用，不能代表真实精度**。

重新训练：

```bash
cd ~/underwater_slam_project
python3 scripts/train_yolo.py --epochs 100 --imgsz 640 --batch 4 --device 0
python3 scripts/export_yolo_onnx.py --weights models/best.pt --imgsz 640
```

训练数据在：

```text
datasets/raw/rubbish.v1i.yolov8/
datasets/rubbish_yolov8_prepared/
```

---

## 8. 当前算法包含什么

`image_enhancer_cpp` 处理顺序：

```text
输入 BGR
  -> 去畸变（当前开启）
  -> 红通道补偿
  -> Gray World 白平衡
  -> LAB L 通道 CLAHE
  -> 可选锐化（当前关闭）
  -> /image_enhanced
```

ORB-SLAM3 本身不做白平衡/CLAHE，它使用增强后的图像。  
图像增强配置在：

```text
config/enhancer_underwater.yaml
src/underwater_perception/config/enhancer_underwater.yaml
```

---

## 9. 重要说明与限制

1. **地图是稀疏的**：ORB-SLAM3 输出相机轨迹 + 稀疏点云，不是稠密 Mesh/房间模型。
2. **单目没有绝对尺度**：`path length` 是地图单位，不是米。
3. **必须平移**：纯旋转不能初始化；速度要慢。
4. **同一时间只能跑一套 mapping**：否则两个进程会抢 UVC 相机，导致设备重新枚举、读帧失败。
5. **相机编号会变**：用 `/dev/v4l/by-id/...`，不要用 `/dev/video2/3`。
6. **viewer 退出用 Ctrl+C**，不要 Ctrl+Z（会挂起）。
7. 1280x720 当前约 6~8 fps；若跟踪不稳，用 640x480，但必须在 640x480 下重新标定。

---

## 10. 文档索引

```text
docs/RUNBOOK.md              # 启动、排障
docs/CALIBRATION.md          # 标定流程与判读
docs/YOLO_REAL_DATA.md       # 真实环境 YOLO 数据/训练/接入
docs/INTERFACES.md           # 话题、消息、参数契约
docs/思路报告.md              # 算法审查、问题清单、改进路线、PPT 素材
docs/AI_HANDOFF.md           # 给 AI/接手的工程说明
docs/UNVERIFIED.md           # 已验证/未验证边界
docs/实际操作步骤.md          # 原操作步骤
```
