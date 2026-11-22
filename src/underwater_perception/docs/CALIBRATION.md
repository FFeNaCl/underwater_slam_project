# 相机标定指南（水下）

## 1. 为什么必须自己标定
项目里所有几何量都依赖内参：单目测距 `Z = f·L/s`、去畸变、三角化射线、ORB-SLAM3 初始化。
用错内参的典型症状：距离整体偏大/偏小、SLAM 初始化困难、图像边缘的特征点被错误投影。

**水下还要额外注意**：平面防水壳引入"空气-玻璃-水"多介质折射，
普通针孔 + 径向畸变模型只是近似，越靠画面边缘误差越大。因此：
* 标定与使用时"防水壳、水深、镜头到玻璃的距离"要一致；改变后需重标。
* 不要为了刷低 RMS 而只用画面中心的小范围视图——那样边缘误差会留在使用阶段。
* 想做高精度需要球面端口或折射模型，本项目不实现（在报告里如实说明即可）。

## 2. 采集要求
| 项目 | 要求 |
|---|---|
| 图案 | 棋盘格 **8x6 内角点**（= 9x7 个方格），方格边长 **25 mm**（`--square 0.025`） |
| 数量 | ≥12 个有效视图（脚本会警告），推荐 **20~30** 个 |
| 姿态覆盖 | 画面中心区 + 四个角、正向 + 左右/上下倾斜（±20°~40°） |
| 环境 | 相机与标定板**都在水下**，水质/光照接近实际实验条件 |
| 图像 | 用**增强后**的图像（`/image_enhanced`），与实际使用链路一致；避免运动模糊 |
| 板子 | 平整（打印后贴硬板）、无强反光、不翘 |

## 3. 四种标定方式

### 3.0 图形界面（标定板参数可直接改，推荐现场使用）

```bash
python3 -m underwater_perception.tools.calibrate_gui
# 或
ros2 run underwater_perception calibrate_gui
```

界面左侧可改：

* 图案类型（棋盘格 / 圆点阵列）；
* **内角点列数 / 行数**；
* **方格边长（毫米）**；
* 最少/最多视图、采集目标、自动采集间隔；
* 固定 k3、固定宽高比、切向畸变置零、去畸变 alpha；
* USB 相机序号/分辨率、视频文件、图片文件夹；
* 输出目录、相机名称、ORB-SLAM3 设置名、Camera.fps。

改参数时下方会实时显示“内角点 8x6；方格 9x7；物理尺寸约 225 x 175 mm”，
可当场发现“把 9x7 个方格误填成 9x7 内角点”这类错误。

实时采集窗口按键：

| 按键 | 作用 |
|---|---|
| `SPACE` | 采集当前帧（标定板不完整时不采） |
| `A` | 自动采集开/关（按面板间隔，且会跳过过于相似的视图） |
| `R` | 清空已采集视图 |
| `C` | 结束采集并立即标定 |
| `Q` / `ESC` | 放弃本次采集 |

标定完成后自动写出四件套，并保存 5 张去畸变示例图。
没有图形界面（SSH/无显示器）时请用下面的 CLI 或 ROS 节点。

### 3.1 离线图片（推荐：可重标、可复现）
```bash
python3 -m underwater_perception.tools.calibrate_cli \
  --images ~/calib_images --cols 8 --rows 6 --square 0.025 \
  --min-views 12 --out ~/underwater_slam_project/calibration_out
```
先看有没有采够：
```bash
# 打印生效参数（确认 cols/rows/square 真的传进去了）
python3 -m underwater_perception.tools.calibrate_cli --dump-config
```

### 3.2 实时相机（交互式，空格采集、q 结束）
```bash
python3 -m underwater_perception.tools.calibrate_cli --camera 0 --capture 25 \
  --fourcc MJPG --camera-fps 30
```

> **相机序号别选错**：Linux 下 UVC 相机常同时创建取流节点和 metadata 节点
> （例如 `/dev/video0` 取流、`/dev/video1` metadata）。`/dev/video1` 打不开或没有图像。
> 先 `v4l2-ctl --list-devices` 确认，通常用 `--camera 0`（第二台相机是 `2`）。
>
> 1280x720 下 `YUYV` 未压缩，USB2 相机通常只有 5~10 fps；**必须用 MJPG** 才能到 30 fps。
> 工具默认已经是 `--fourcc MJPG`，并会打印实际协商到的格式/帧率。

### 3.3 ROS 节点（边看调试图边采）
```bash
ros2 launch underwater_perception calibration.launch.py image_topic:=/image_enhanced
ros2 run rqt_image_view rqt_image_view          # 看 /calibration/debug
ros2 topic echo /calibration/status --once      # 看 views / rejected
ros2 service call /calibration_node/run std_srvs/srv/Trigger "{}"
```
采集是**自动**的（默认每 1.0 s 尝试一帧，`capture_interval_s` 可调），
也可以随时调用 `~/capture` 手动补一帧。`~/reset` 清空重来。

## 4. 输出文件（四件套）与怎么用

| 文件 | 用途 |
|---|---|
| `ost.yaml` | ROS `camera_calibration` 兼容格式（`camera_matrix`/`distortion_coefficients`/`projection_matrix`），别的节点也能读 |
| `water_1280x480.yaml` | **ORB-SLAM3 设置文件**，复制到 `orb_slam3/config/Monocular/`；`Camera.RGB: 0` 对应本项目的 BGR 图像链路 |
| `enhancer_camera_params.yaml` | 图像增强节点的 `camera_matrix`(9 个数) 与 `distortion_coeffs`(5 个数) |
| `calibration_report.json` | 全部指标、每个视图的检出与误差、警告、OpenCV 版本（写报告用） |
| `undistorted/` | 去畸变示例图（人眼检查边缘是否变直） |

安装到链路：
```bash
cp calibration_out/water_1280x480.yaml \
   ~/underwater_slam_project/src/ros2_orb_slam3/orb_slam3/config/Monocular/
# 然后把 enhancer_camera_params.yaml 里的两个数组填进 config/enhancer_underwater.yaml，
# 并把 use_undistort 改成 true，重启增强节点
# 注意：这两个键默认是**注释掉**的。ROS2 把空数组 [] 当作“未设置”，
# C++ 节点会抛 InvalidParameterValueException；标定后请取消注释再填数。
```

## 5. 结果判读（**别只看 RMS**）

| 指标 | 好的范围 | 偏大意味着什么 |
|---|---|---|
| RMS 重投影误差 | < 0.5 px 优秀；0.5~1.0 可接受 | >1.0：角点检测差（模糊/反光/板不平）、视图太少、畸变模型不合适 |
| fx/fy 一致性 | 两者相差 < 2% | 相差大：`square_size` 写错、宽高比参数有问题 |
| 主点位置 | 应在图像中心附近（±几十 px） | 明显偏离：视图姿态太单一（都在同一个位置） |
| 逐视图误差 | 最大不超过均值的 3 倍 | 个别视图异常：剔除后重标（脚本会点名） |
| 覆盖度警告 | 无 | 角点只集中在画面一小块 → 边缘参数不可信 |
| 去畸变效果 | 直线变直、无波浪 | 仍有弯曲：模型不足（多介质折射），需在报告中说明 |

脚本会明确列出"误差最大的 3 个视图"，把可疑图片删掉再跑一次通常能显著改善。

## 6. 常见失败与处理

| 症状 | 原因 | 处理 |
|---|---|---|
| `pattern_not_found` 很多 | 板子不完整、曝光过度、运动模糊 | 让板子完整入画、避免过曝、静止拍摄 |
| 有效视图 < 3，直接失败 | 图案参数不符 | 确认 `cols/rows` 是**内角点**数、`square_size` 单位是米 |
| `grid_check_failed` | 检测器锁到偏移/外扩点阵 | 画面里只放一块板子、避免背景里有类似网格（瓷砖、栏杆） |
| 标定成功但测距明显偏差 | `square_size` 与实际不符 | 用卡尺量方格边长（打印缩放会改尺寸！） |
| ORB-SLAM3 用标定结果后更差 | 标定图与实际使用环境不一致（壳/水深变了） | 在真实使用条件下重标 |
| 视图尺寸不一致被忽略 | 混用了不同分辨率的图 | 统一分辨率后重采 |

## 7. 验收清单（做到这几条才算真的标定好了）
- [ ] 有效视图 ≥ 20，且角点覆盖画面中心与四角
- [ ] RMS < 0.5 px，无"覆盖度不足"警告
- [ ] 逐视图误差没有明显离群
- [ ] 去畸变示例图里直线确实变直（打开 `undistorted/` 看图）
- [ ] `ost.yaml` 的 `camera_matrix` 与 `enhancer_camera_params.yaml` 数值一致
- [ ] ORB-SLAM3 用的 yaml 与上面同源（同一次标定的产物）
- [ ] 报告 JSON 已留档（写结题材料时直接引用）
