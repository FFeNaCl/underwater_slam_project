# 接口参考（INTERFACES）

本文件是"查表用"的契约汇总：话题、消息、参数。**改动接口时必须同步更新这里**，
否则接手的人（或 AI）会以过期信息为依据。

约定：`sensor_msgs/Image` 一律按 `(H, W, C)`；像素坐标 `(u, v)` 原点左上；
时间戳 `sec + nanosec`；角度弧度；长度米（除单目 SLAM 地图尺度外，见下）。

---

## 1. 话题

### 1.1 图像与检测

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | 相机/播放器 → 增强 | 原始图像 |
| `/image_enhanced` | `sensor_msgs/Image` | 增强 → 检测/桥接 | 增强后图像（推荐主输入） |
| `/image_enhanced_debug` | `sensor_msgs/Image` | 增强 → 人看 | 原图/增强图拼接（`publish_debug_view` 控制） |
| `/detections` | `std_msgs/String` | 检测 → 地标/记录 | JSON，契约见 §2 |
| `/detections/annotated` | `sensor_msgs/Image` | 检测 → 人看 | 轮廓+指标标注图 |
| `/detections/markers` | `visualization_msgs/MarkerArray` | 检测 → RViz | 相机系方块+文字 |
| `/yolo/annotated` | `sensor_msgs/Image` | YOLO → 人看 | 既有节点 |
| `/yolo/detections` | `std_msgs/String` | YOLO → 地标 | 与 `/detections` 同契约 |

### 1.2 ORB-SLAM3 输入（**话题名在 C++ 里硬编码，勿改**）

| 话题 | 类型 | 说明 |
|---|---|---|
| `/mono_py_driver/img_msg` | `sensor_msgs/Image` | 图像（建议 `bgr8`） |
| `/mono_py_driver/timestep_msg` | `std_msgs/Float64` | 秒；**必须严格递增**。2026-11-19 起 C++ 节点改用图像 `header.stamp` 作为 SLAM 时间基准（避免跨话题乱序），该话题保留兼容旧 driver |
| `/mono_py_driver/experiment_settings` | `std_msgs/String` | 发送设置文件名（不含 `.yaml`） |
| `/mono_py_driver/exp_settings_ack` | `std_msgs/String` | 内容固定为 `ACK` |

### 1.3 SLAM 输出（本次补丁新增）

| 话题 | 类型 | 频率 | 说明 |
|---|---|---|---|
| `/slam/pose` | `geometry_msgs/PoseStamped` | 每帧 | **Twc（世界<-相机）**，`frame_id=map` |
| `/slam/path` | `nav_msgs/Path` | 每帧 | 累积轨迹（Twc），长度上限 `max_path_length` |
| `/slam/map_points` | `sensor_msgs/PointCloud2` | 默认 1 Hz | 字段 `x,y,z` FLOAT32，`point_step=12` |
| `/slam/status` | `std_msgs/String` | 每帧 | JSON：`tracking_state`、`tracking_state_code`、`n_map_points`、`image_scale`、`fps` |
| TF `map -> camera` | `tf2_msgs/TFMessage` | 每帧 | 与 `/slam/pose` 同一变换 |

`tracking_state` 取值：`SYSTEM_NOT_READY`(-1)、`NO_IMAGES_YET`(0)、`NOT_INITIALIZED`(1)、
`OK`(2)、`RECENTLY_LOST`(3)、`LOST`(4)、`OK_KLT`(5)。

⚠️ `LOST` 时节点**仍会发布最后位姿**（避免 RViz 里相机突然消失）。
消费方应通过 `/slam/status` 判断位姿是否可信，而不是假设"有 pose 就是好的"。

### 1.4 语义地标

| 话题 | 类型 | 说明 |
|---|---|---|
| `/semantic/landmarks` | `std_msgs/String` | JSON：`count` + `landmarks[]`（见 §3） |
| `/semantic/markers` | `visualization_msgs/MarkerArray` | 立方体 + 文字；**绿=triangulated，黄=scale_fallback** |

### 1.5 标定

| 话题/服务 | 类型 | 说明 |
|---|---|---|
| `/calibration/debug` | `sensor_msgs/Image` | 画出检出角点的调试图 |
| `/calibration/status` | `std_msgs/String` | JSON：`views`、`rejected`、`ready`、`pattern_found` |
| `/calibration_node/capture` | `std_srvs/Trigger` | 手动采一帧（需当前帧检出棋盘） |
| `/calibration_node/run` | `std_srvs/Trigger` | 执行标定并写出四件套 |
| `/calibration_node/reset` | `std_srvs/Trigger` | 清空已采集视图 |

---

## 2. `/detections` JSON（schema_version = 1）

顶层：`schema_version`、`stamp{sec,nanosec}`、`frame_id`、`source`、`count`、
`image{width,height}`、`detections[]`、`meta{}`。

`detections[i]`：
* 必有：`class_id`(int)、`class_name`(str)、`confidence`(float 0~1)、`xyxy`([x1,y1,x2,y2] float)
* 可选：`track_id`(int)、`detector`(str)、`score_terms`(dict[str,float])
* `geometry{}`：`center`、`corners`、`vertex_count`、`area_px`、`hull_area_px`、`fill_ratio`、
  `convexity`、`extent`、`aspect_ratio`、`angle_deg`、`side_px`、`long_side_px`、
  `edge_direction_count`、`interior_edge_count`、`edge_support`、`internal_corner_response`、
  `color_purity`、`mean_hue`、`mean_saturation`、`mean_value`
* `measurement{}`：`method`、`quality`、`distance_m`、`bearing_rad[2]`、`ray_camera[3]`、
  `pose_rvec[3]`、`pose_tvec[3]`、`fit_error_px`

`measurement.method` 语义表（**决定字段能不能信**）：

| method | 含义 | `distance_m` 可信度 |
|---|---|---|
| `known_size_pose` | 4 角点 `solvePnP(IPPE_SQUARE)` | 高（正视） |
| `known_size_refined` | 轮廓拟合（ICP+PnP，任意视角） | 高（实测 <1% 合成误差） |
| `known_size` | 正视近似 `f·L/s` | 中（斜视有 ~25% 系统误差） |
| `bearing_only` | 只有方向 | 无（不含距离） |
| `none` | 缺内参 | 无 |

**兼容性承诺**：字段只增不改，语义不变；新增字段必须放 `geometry{}`/`measurement{}`/`meta{}` 内。

`/yolo/detections` 复用同一契约，额外带 `model_loaded`(bool)：
`source` 为 `model`（正常推理）、`label_replay`（数据集回放）或
`model_unavailable`（`require_model=true` 但 `model_path` 没填/文件不存在）。
只有 `source=model` 时检测框才代表真实模型输出；`model_unavailable` 时 `count` 必为 0。

---

## 3. `/semantic/landmarks` JSON

```json
{"count": 2, "landmarks": [{
   "landmark_id": 1, "track_id": 3, "class_name": "cube", "frame_id": "map",
   "position": [0.31, -0.02, 1.42],
   "n_views": 7, "reproj_error_px": 1.8, "triangulation_error": 0.004,
   "source": "triangulated", "confidence": 0.87,
   "first_stamp": 1730000000.1, "last_stamp": 1730000002.4, "distance_m": 1.44
}]}
```

| source | 参考系 | 尺度 | 用途 |
|---|---|---|---|
| `triangulated` | `map`（SLAM 地图） | 与点云/轨迹一致（**非米**） | 建图结果、答辩展示 |
| `scale_fallback` | 相机光学系（检测消息的 `frame_id`） | 米（依赖已知边长） | SLAM 未初始化时的临时提示 |

---

## 4. 参数速查（键名 = YAML 键名 = `-p` 名）

| 节点 | 配置文件 | 关键参数 |
|---|---|---|
| `cube_detector` | `config/cube_detector.yaml` | `color_preset`、`hsv_ranges`(6N int)、`mode`、`min_area_px`、`min_fill_ratio`、`min_convexity`、`min_confidence`、`use_silhouette_fit`、`cube_edge_m`、`camera_f*`、`enable_tracking`、`track_min_hits`、`max_rate_hz`、`input_topic`、`detections_topic` |
| `slam_input_bridge` | `config/slam_bridge.yaml` | `image_topic`、`target_*`(**勿改**)、`settings_name`、`use_header_stamp`、`min_dt_s`、`max_rate_hz`、`handshake_retry_s`、`rehandshake_period_s`、`output_encoding` |
| `semantic_landmark` | `config/semantic_landmark.yaml` | `min_views`、`min_parallax_deg`、`min_baseline_m`、`max_reproj_error_px`、`stale_after_s`、`allow_scale_fallback`、`cube_edge_m`、`landmark_class_names`、`min_detection_confidence`、`camera_f*`、`publish_period_s`、`max_pose_age_s` |
| `calibration_node` | `config/calibration.yaml` | `pattern`、`cols`、`rows`、`square_size`、`min_views`、`max_views`、`min_corners_ratio`、`subpix_window`、`fix_k3`、`undistort_alpha`、`output_dir`、`orbslam3_settings_name`、`auto_capture`、`capture_interval_s` |
| `image_sequence_player` | — | `topic`、`synthetic`、`image_dir`、`fps`、`loop`、`width`、`height`、`cube_edge_m`、`distance_m`、`num_frames` |
| `dataset_capture` | `config/dataset_capture.yaml` | `image_topic`、`output_dir`、`save_interval_s`、`max_images`、`image_format`、`write_empty_labels`、`enabled`、`min_width`、`min_height`、`status_topic` |
| `camera` | — | `camera_index`、`width`、`height`、`fourcc`、`fps`、`topic`、`frame_id`、`log_interval_s` |
| `slam_viewer` | — | `path_topic`、`points_topic`、`status_topic`、`landmarks_topic`、`image_topic`、`window_name`、`update_hz`、`max_points` |
| `yolo_detector`(既有，已扩展) | `config/yolo_real_camera.yaml` | `model_path`(**best.pt 路径**)、`require_model`、`confidence_threshold`、`imgsz`、`device`、`half`、`max_det`、`input_topic`、`detections_topic` |
| `ros2_orb_slam3/mono_node_cpp` | 启动参数 | `voc_file_arg`、`settings_file_path_arg`(**目录**)、`node_name_arg`、`enable_pangolin`(默认 false)、`publish_*`、`map_frame_id`、`camera_frame_id`、`max_path_length`、`point_publish_period_ms` |
| `image_enhancer_cpp`(既有) | `config/enhancer_underwater.yaml` | `use_red_compensation`、`use_gray_world`、`use_clahe`、`clahe_clip_limit`、`use_undistort`、`camera_matrix`(9)、`distortion_coeffs`(5) |

**参数声明规则**：ROS 参数只支持 bool/int/double/string 及其**同构数组**，
没有"数组的数组"。因此 HSV 区间用扁平整型数组（6 个一组），
`dist_coeffs` 用 5 个 double。键名错写不会报错、只会静默失效 ——
`test/test_configs.py` 会静态拦住这种情况。
