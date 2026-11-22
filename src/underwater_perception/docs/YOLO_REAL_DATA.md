# 真实环境 YOLO 数据集：采集、标注、训练、接入

> 适用场景：**真实水下相机**（水池/水箱/泳池/海试），不是 VRX/Gazebo 仿真。
> 目标不是“识别出任何东西”，而是训练出**能和单目 SLAM 配合的静态语义目标检测器**。
>
> 本项目的部署链路是：
> `/image_raw -> image_enhancer_cpp -> /image_enhanced -> yolo_detector -> /yolo/detections`
> 因此**训练数据的图像分布要尽量和部署时的 `/image_enhanced` 一致**。

---

## 1. 先定类别：不是越多越好

### 1.1 两类目标必须分开

| 目标类型 | 例子 | 在 SLAM 里怎么用 | 是否适合做语义地标 |
|---|---|---|---|
| **静态环境目标** | 红色标定立方体、标记板、管道、阀门、沉底箱、门架 | 多视图三角化 -> 地图中的三维点 | ✅ 适合 |
| **动态/干扰目标** | 鱼、气泡、漂浮物、潜水员、水面反光 | 转成 mask，在 ORB 特征提取时剔除 | ❌ 不适合（位置会变） |

> **重要**：`semantic_landmark` 节点默认会把所有检测都尝试三角化。
> 如果数据集里同时有“鱼”和“立方体”，请先只把静态类接到 `/yolo/detections`，
> 或后续给地标节点加类别白名单（见第 9 节“下一步改造”）。

### 1.2 建议的类别设计（按你的实验目标选一套）

**方案 A：几何标定物（当前代码最容易验证）**

```text
cube        # 红色小立方体（已知边长 5 cm）
marker      # 平面标记板 / ArUco / 棋盘格板（静态）
pipe        # 管道
valve       # 阀门
```

**方案 B：水下场景语义目标**

```text
buoy / marker / pipe / valve / gate / debris
```

**方案 C：动态干扰物（只用于 mask，不做地标）**

```text
fish / bubble / diver / floating_debris
```

建议第一版 **3~5 类**，每类至少 **200 个实例**再做训练；类别边界写进标注规范，不要边标边改。

---

## 2. 采什么：场景矩阵比“数量”更重要

每个类别尽量覆盖下面的组合，避免模型只记住一种水质/距离/角度：

| 维度 | 建议覆盖 |
|---|---|
| 水质 | 清水、微浑、浑水（能见度 0.5 / 1 / 2 m） |
| 光照 | 自然光、补光灯、逆光、水面反光 |
| 距离 | 近（0.3 m）、中（0.8 m）、远（1.5~2 m） |
| 视角 | 正视、左右斜视 ±30°、俯仰 ±20° |
| 遮挡 | 目标被水草/管道/气泡挡住 20%~50% |
| 运动 | 静止、缓慢平移（模拟 SLAM 运动） |
| 负样本 | 纯水、池底瓷砖、红色杂物、反光斑、无目标画面 |

采集时的三个硬要求：

1. **训练/部署同源**：训练用 `/image_enhanced`，部署也走 `/image_enhanced`；增强参数要固定并记录。
2. **不要连续帧全收**：默认每 0.5 s 一帧；同一姿势不要超过 3~5 张，否则数据集高度冗余。
3. **按场次记录**：每次下水单独一个目录/前缀，后面按场次拆分训练/验证，避免泄漏。

---

## 3. 怎么采：两种工具

### 3.1 ROS 节点（推荐：相机已经通过 ROS 发布）

```bash
# 终端 1：相机（本项目 OpenCV 节点，MJPG；video1/video3 是 metadata，别用）
source /opt/ros/humble/setup.bash
source /home/hu/underwater_slam_project/install/setup.bash
ros2 launch underwater_perception camera.launch.py \
  camera_index:=0 width:=1280 height:=720 fourcc:=MJPG fps:=30.0

# 终端 2：增强（训练/部署一致）
ros2 run image_enhancer_cpp enhancer --ros-args \
  --params-file /home/hu/underwater_slam_project/config/enhancer_underwater.yaml

# 终端 3：采集
source /home/hu/underwater_slam_project/install/setup.bash
ros2 launch underwater_perception dataset_capture.launch.py \
  image_topic:=/image_enhanced \
  output_dir:=/home/hu/underwater_slam_project/datasets/real_underwater/session_01/raw \
  save_interval:=0.5 \
  max_images:=600
```

服务控制：

```bash
ros2 service call /dataset_capture/stop std_srvs/srv/Trigger "{}"
ros2 service call /dataset_capture/start std_srvs/srv/Trigger "{}"
ros2 service call /dataset_capture/snapshot std_srvs/srv/Trigger "{}"   # 补拍当前帧
ros2 topic echo /dataset_capture/status --once
```

> 如果也想保留原图用于以后换增强算法，可以再开一个采集节点：
> `image_topic:=/image_raw output_dir:=.../session_01/raw_original`。

### 3.2 独立工具（没有 ROS / 直接插 USB 相机）

```bash
cd /home/hu/underwater_slam_project
python3 -m underwater_perception.tools.capture_dataset \
  --camera 0 \
  --out ~/underwater_slam_project/datasets/real_underwater/session_01/raw \
  --interval 0.5 --max 600 \
  --width 1280 --height 720 \
  --fourcc MJPG --fps 30 \
  --names cube,marker,pipe
```

> `/dev/video1`、`/dev/video3` 通常是 UVC metadata 节点，打不开或不出图；
> 用 `v4l2-ctl --list-devices` 确认取流节点（一般是 video0 / video2）。
> 1280x720 下 YUYV 在 USB2 上只有 5~10 fps，**必须 MJPG**。

按键：`SPACE` 保存、`A` 自动采集开关、`Q/ESC` 退出。
没有显示器时加 `--no-display`（自动采集，到达 `--max` 自动结束）。

视频抽帧：

```bash
python3 -m underwater_perception.tools.capture_dataset \
  --video pool.mp4 --no-display --interval 0.5 --max 800 \
  --out ~/underwater_slam_project/datasets/real_underwater/session_02/raw
```

### 3.3 输出结构

```text
session_01/raw/
├── images/                  # 原始帧，文件名 frame_000123_<纳秒>.jpg
├── labels/                  # --write-empty-labels 时生成空 txt
├── data.yaml                # 给了 --names 时生成模板（标注后要改路径）
└── capture_session.json     # 采集时间、来源、数量、间隔 -> 写报告用
```

---

## 4. 怎么标：YOLO 格式

推荐工具：

* **X-AnyLabeling**（本地免费，支持 YOLOv8 自动预标注，人工修正最快）；
* **labelImg**（经典、纯手动，适合小数据集）；
* **Roboflow**（在线，团队协作方便）。

导出结构（**训练前必须按场次拆成 train/valid，不要用同一批图**）：

```text
my_dataset/
├── data.yaml
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
└── valid/
    ├── images/*.jpg
    └── labels/*.txt
```

标签格式（每行一个目标，全部归一化到 0~1）：

```text
class_id cx cy w h
```

标注规范：

* **负样本**：没有目标的图保留**空 txt**（不是删除 txt）；
* **小目标**：远处目标也要框，bbox 尽量贴紧目标外轮廓；
* **遮挡**：框可见部分，不要脑补被挡住的部分；
* **一致性**：同一物体在不同帧里类别不能变；
* **不要框动态模糊到无法辨认的目标**（模型学不到稳定特征）。

---

## 5. 怎么拆：按场次，不按帧

最致命的错误是“把所有帧随机打乱后 8:2 拆分”。相邻帧几乎一样，验证集等于训练集，
指标虚高，真实下水就崩。

正确做法：

```text
session_01（清水） -> train
session_02（微浑） -> train
session_03（浑水） -> valid
session_04（新场景） -> test
```

比例建议 **70% / 20% / 10%**，且每个 split 都要有各类别和负样本。
如果只有一次拍摄，至少按“时间段”切开：前 70% 训练、后 30% 验证。

---

## 6. 训练前体检

```bash
cd /home/hu/underwater_slam_project
python3 -m underwater_perception.tools.yolo_dataset_check \
  --dataset ~/underwater_slam_project/datasets/real_underwater \
  --json ~/underwater_slam_project/datasets/real_underwater/check_report.json
```

会检查：

* 图片/标签一一对应、孤立标签；
* 标签行是否合法（归一化、宽高 >0、class_id 越界）；
* 类别实例数是否严重不均衡；
* **跨 train/valid 的重复图片（泄漏）**；
* 图像尺寸分布。

生成 `data.yaml`：

```bash
python3 -m underwater_perception.tools.yolo_dataset_check \
  --dataset ~/underwater_slam_project/datasets/real_underwater \
  --names cube,marker,pipe \
  --write-data-yaml
```

---

## 7. 训练与导出

### 7.1 一键脚本（推荐）

```bash
cd /home/hu/underwater_slam_project

# 1) 整理数据集 + 训练：自动按时间切出 valid，完成后复制 models/best.pt
python3 scripts/train_yolo.py \
  --data /home/hu/Downloads/rubbish.v1i.yolov8 \
  --epochs 100 --imgsz 640 --batch 4 --device 0

# 2) 导出 ONNX：旧 models/best.onnx 自动备份到 models/backup/
python3 scripts/export_yolo_onnx.py \
  --weights /home/hu/underwater_slam_project/models/best.pt --imgsz 640
```

脚本行为：

* 原始下载目录**只读**，整理后的数据写到 `datasets/rubbish_yolov8_prepared/`；
* 源数据只有 `train` 时，按文件名时间切最后 `--val-ratio`（默认 20%）做 valid；
* 训练结果写到 `runs/yolo/<name>/`，默认复制 `best.pt` 到 `models/best.pt`；
* 导出前先备份旧的 `models/best.onnx`，再用 onnxruntime 自检输入/输出形状。

> **验证集很小时加 `--copy-last`**：Ultralytics 按验证集选 `best.pt`，
> 3 张验证图容易选到“置信度≈0”的退化模型（本项目实际遇到过）。
> 数据量 <50 张时建议：
> ```bash
> python3 scripts/train_yolo.py ... --copy-last
> ```

### 7.2 手动命令

```bash
cd /home/hu/underwater_slam_project
mkdir -p runs/yolo

# 先轻量模型保实时，效果不够再换 yolov8s / yolov11s
yolo detect train \
  data=/home/hu/underwater_slam_project/datasets/real_underwater/data.yaml \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  patience=20 \
  project=/home/hu/underwater_slam_project/runs/yolo \
  name=underwater_real_v1
```

必看指标（不要只看训练 loss）：

* `results.png`：mAP50 / mAP50-95 曲线；
* `confusion_matrix.png`：哪两类互相混淆；
* `val_batch*.jpg`：验证集预测可视化；
* 每类 **Recall**：小目标/遮挡类单独看；
* 目标机上的 **FPS**（`device=cpu` 和 GPU 各测一次）。

导出 ONNX（可选，CPU 部署更稳）：

```bash
yolo export \
  model=/home/hu/underwater_slam_project/runs/yolo/underwater_real_v1/weights/best.pt \
  format=onnx imgsz=640
```

---

## 8. 放进程序：`best.pt` 位置留空，训练后填

### 8.1 只需要改一处

打开 `/home/hu/underwater_slam_project/config/yolo_real_camera.yaml`：

```yaml
yolo_detector:
  ros__parameters:
    model_path: ""       # TODO: 训练完成后填绝对路径
```

训练完成后改成：

```yaml
    model_path: /home/hu/underwater_slam_project/runs/yolo/underwater_real_v1/weights/best.pt
```

> 仓库里**没有**放 `best.pt` 占位文件，避免误以为“已经有可用模型”。
> 在填之前节点会显示 `MODEL NOT LOADED - set model_path to best.pt`，
> 并在 `/yolo/detections` 里给出 `source: "model_unavailable"`、`count: 0`，
> **不会**用标签回放冒充检测结果。

### 8.2 启动真实相机 YOLO 链路

```bash
source /home/hu/underwater_slam_project/install/setup.bash

# 相机自己在终端 1 启动（见 3.1）
ros2 launch underwater_perception yolo_real.launch.py \
  model_path:=/home/hu/underwater_slam_project/runs/yolo/underwater_real_v1/weights/best.pt \
  device:=cpu
```

查看：

```bash
ros2 topic hz /image_enhanced
ros2 topic hz /yolo/annotated
ros2 topic echo /yolo/detections --once
ros2 run rqt_image_view rqt_image_view    # 选 /yolo/annotated
```

### 8.3 接进 SLAM 的语义地标

`semantic_landmark` 默认订阅 `/detections`（几何旁路）。使用 YOLO 时改一处：

```bash
ros2 run underwater_perception semantic_landmark --ros-args \
  -p detections_topic:=/yolo/detections
```

或修改 `config/semantic_landmark.yaml` 后重启。只把**静态目标**接进来；
鱼/气泡等动态目标不要做地标。建议同时设置类别白名单：

```yaml
landmark_class_names: ["cube", "marker", "pipe"]
min_detection_confidence: 0.35
```

这样即使 YOLO 同时输出 `fish/bubble`，它们也不会被三角化进地图。

---

## 9. 程序改造思路（为什么这么改）

### 9.1 检测节点（`yolo_vrx_bridge/yolo_detector_node.py`）

新增参数：

| 参数 | 作用 |
|---|---|
| `model_path` | `.pt` 或 `.onnx` 路径；**真实配置里留空占位** |
| `require_model` | 真实环境设为 `true`：模型没加载时明确报警，不回放标签 |
| `imgsz` | 推理尺寸，训练/部署保持一致（默认 640） |
| `device` | `""` 自动、`cpu`、`0`（NVIDIA） |
| `half` | CUDA 上半精度；CPU 不要开 |
| `max_det` | 单帧最多检测框数 |
| `allow_label_replay` | 真实环境设为 `false` |

输出 JSON 增加 `model_loaded`，`source` 取值：

* `model`：真实模型推理；
* `label_replay`：数据集回放（只用于演示/离线验证）；
* `model_unavailable`：要求模型但没加载（`count=0`）。

### 9.2 数据采集节点（`underwater_perception/nodes/dataset_capture_node.py`）

* 只做“订阅图像 + 按间隔落盘”，不碰检测算法；
* 与独立工具共用 `core/dataset_io.py`，文件名/编码/中文路径行为一致；
* 服务化控制（start/stop/snapshot/reset），方便现场操作；
* 每次采集写 `capture_session.json`，写报告时能追溯数据来源。

### 9.3 数据集检查工具（`tools/yolo_dataset_check.py`）

训练前把“标签错误/类别越界/跨集重复”变成可执行的检查项，
避免把数据问题误判成模型问题。

### 9.4 下一步改造（答辩前可选）

1. ✅ **地标节点类别白名单（已实现）**：`landmark_class_names` + `min_detection_confidence`，
   只允许 `cube/marker/pipe` 三角化，避免鱼变成地标；
2. **动态 mask 接 SLAM**：把 `fish/bubble/diver` 的框转成 mask，在 ORB 特征提取前剔除；
3. **多帧确认**：YOLO 单帧置信度低的目标，要求连续 N 帧命中才作为地标观测；
4. **类别到地标类型的映射**：不同类别用不同 Marker 颜色/尺寸；
5. **尺度恢复**：用已知尺寸的静态目标（立方体/标记板）给单目 SLAM 提供米制尺度。

---

## 10. 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| 画面上一直是 `MODEL NOT LOADED` | `model_path` 空/路径错 | 填绝对路径，`ls` 确认文件存在 |
| 检测框类别名不对 | 模型元数据类别与数据集不一致 | 检查 `data.yaml` 的 `names` 顺序；必要时设 `dataset_root` |
| 训练 mAP 很高，下水就崩 | 训练/验证同源、相邻帧泄漏 | 按场次拆分，跑 `yolo_dataset_check` 看跨集重复 |
| 水下漏检严重 | 训练图多为水面/清水、目标小 | 补浑水、远距离、遮挡样本；提高输入尺寸 |
| 检测到鱼也变成地标 | 动态类也接到了地标节点 | 只接静态类，或加类别白名单 |
| CPU 上 FPS 太低 | `imgsz=640` + 大模型 | 用 `yolov8n`、`imgsz=480/512`、`device=cpu` 实测 |
| 增强后颜色变了，模型不认 | 训练用原图、部署用增强图 | 训练/部署统一到 `/image_enhanced`，冻结增强参数 |

---

## 11. 写报告/PPT 建议记录的指标

* 数据集：每类实例数、场景数、train/valid/test 划分、负样本比例；
* 训练：mAP50、mAP50-95、每类 Recall、混淆矩阵、训练曲线；
* 部署：CPU/GPU 上的 FPS、单帧延迟、模型大小；
* 真实场景：清水/浑水召回率对比、不同距离检出率、误检类型；
* 与几何旁路对比：同一批真实图片上的检出率/误检数/耗时；
* SLAM 联动：开/关动态 mask 的特征点数、跟踪丢失次数、轨迹漂移。
