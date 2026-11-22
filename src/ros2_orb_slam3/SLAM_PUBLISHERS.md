# ROS2 ORB-SLAM3 单目节点：SLAM 轨迹 / 点云 / 状态发布接口

本文档说明对 `ros2_orb_slam3` 包中 `MonocularMode` 节点（`mono_node_cpp`）所做的一次改动：
**把 ORB-SLAM3 算出来的位姿真正发布到 ROS 上**，并新增点云、状态、TF 等标准接口，
同时修掉几个会导致节点崩溃 / 无显示器起不来的问题。

> 面向的读者是**没有参与本次对话的人（或 AI）**。因此关键决策都写了「为什么」，
> 且明确指出哪些部分**没有在目标机上验证过**（因为改动是在 Windows 上做的，没有 ROS2 环境可编译）。
> 2026-09-10 在目标机上做过一轮 API 级核对与路径适配，结论见 §10；**编译与运行仍未执行**。

---

## 1. 改动前的现状（为什么要改）

`MonocularMode`（`include/ros2_orb_slam3/common.hpp` + `src/common.cpp`）原本：

* 订阅 `/mono_py_driver/experiment_settings`(String) 拿到设置文件名 → 初始化 SLAM →
  发布 `/mono_py_driver/exp_settings_ack`(String, 内容 `"ACK"`)。**这个握手逻辑本次完全保留。**
* 订阅 `/mono_py_driver/img_msg`(sensor_msgs/Image) 与 `/mono_py_driver/timestep_msg`(std_msgs/Float64)。
* 在 `Img_callback()` 里调用 `Sophus::SE3f Tcw = pAgent->TrackMonocular(cv_ptr->image, timeStep);`
  然后**把 Tcw 直接丢掉**——ROS 侧因此永远看不到轨迹、点云、跟踪状态。
* `initializeVSLAM()` 里硬编码 `enablePangolinWindow = true; enableOpenCVWindow = true;`
  → 没有显示器 / 没有 OpenGL 上下文（纯 SSH、CI、Docker、systemd 服务）时节点起不来。
* 析构函数无条件 `pAgent->Shutdown();`，而 `pAgent` **在构造函数里没有初始化**（野指针）
  → 没收到设置就退出（Ctrl-C、driver 没起来）会段错误。
* `settingsFilePath = settingsFilePath.append(configString); ... .append(".yaml");`
  就地修改成员，**重复收到 `experiment_settings` 会重复追加**成 `dir/name.yamlname.yaml`。

---

## 2. 改动清单

### 2.1 修改的文件

| 文件 | 改动 |
| --- | --- |
| `include/ros2_orb_slam3/common.hpp` | 新增所需消息头（`geometry_msgs/msg/pose_stamped.hpp`、`nav_msgs/msg/path.hpp`、`sensor_msgs/msg/point_cloud2.hpp`、`sensor_msgs/msg/point_field.hpp`、`tf2_ros/transform_broadcaster.h`、`builtin_interfaces/msg/time.hpp`、`Eigen/Geometry`、`<memory>/<string>/<cstdint>/<cstddef>`、`MapPoint.h`）；`pAgent` 改为 `= nullptr`（**关键**）；新增私有成员：发布器、TF 广播器、参数缓存、轨迹与统计状态；`packagePath` 默认值从 `ros2_test` 改为 `orb_slam3_ws`（目标机路径适配，见 §10.1） |
| `src/common.cpp` | 新增两个文件内静态辅助函数（状态码→字符串、`SE3f`→`Pose`）；构造函数里 `declare_parameter` + 读取 11 个新参数、创建 4 个发布器 + TF 广播器、保存 settings **目录基准**；析构判空；`initializeVSLAM()` 幂等拼接路径、窗口开关改用参数、重复握手先关旧实例；`Img_callback()` 里完成跟踪并发布 5 类输出 |
| `CMakeLists.txt` | `find_package` 增加 `geometry_msgs` / `nav_msgs` / `tf2_ros`，并把三者加入 `THIS_PACKAGE_INCLUDE_DEPENDS`（`mono_node_cpp` 因此自动链接到）。**ORB-SLAM3 库的构建逻辑与可执行目标未做任何改动** |
| `package.xml` | 增加 `<depend>geometry_msgs</depend>` / `<depend>nav_msgs</depend>` / `<depend>tf2_ros</depend>` |
| `scripts/mono_driver_node.py` | 硬编码的 `~/ros2_test/src/ros2_orb_slam3` 数据集路径改为 `~/underwater_slam_project/src/ros2_orb_slam3`（目标机路径适配，见 §10.1）。仅影响 EuRoC 数据集回放模式；真实相机主线由 `underwater_perception` 的 `slam_input_bridge` 供图，不走该节点 |

### 2.2 新增的文件

| 文件 | 说明 |
| --- | --- |
| `SLAM_PUBLISHERS.md` | 本文档 |
| `patches/ros2_orb_slam3_slam_publishers.patch` | 上述五个文件改动的 unified diff（相对**包根目录**），用 `git diff -- <五个文件>` 生成（2026-09-10 重新生成，加入 `scripts/mono_driver_node.py` 的路径适配） |

### 2.3 没有改的东西（有意为之）

* 原有话题名 / 参数名一个都没改：`/mono_py_driver/experiment_settings`、
  `/mono_py_driver/exp_settings_ack`、`/mono_py_driver/img_msg`、
  `/mono_py_driver/timestep_msg`、`node_name_arg`、`voc_file_arg`、`settings_file_path_arg` 全部保持原样。
* 握手时序保持原样：收到设置 → 先发 `"ACK"` → 再 `initializeVSLAM()`。
* 没有新增可执行目标，没有动 `orb_slam3_lib` 的源文件列表。

---

## 3. 新增的接口契约（名字必须一字不差）

### 3.1 话题

| 话题 | 类型 | 内容 |
| --- | --- | --- |
| `/slam/pose` | `geometry_msgs/msg/PoseStamped` | 相机在**世界/地图坐标系**中的位姿 **Twc**（见 §4.1） |
| `/slam/path` | `nav_msgs/msg/Path` | 累积的相机轨迹（同样是 Twc），`header.frame_id = map_frame_id` |
| `/slam/map_points` | `sensor_msgs/msg/PointCloud2` | 来自 `GetAllMapPoints()`，为空时退回 `GetTrackedMapPoints()`；字段 `x,y,z` 全为 FLOAT32，`point_step = 12`，小端 |
| `/slam/status` | `std_msgs/msg/String` | JSON 文本，见 §3.3 |
| `/tf` | `tf2_msgs/msg/TFMessage`（由 `tf2_ros::TransformBroadcaster` 发布） | `map_frame_id` → `camera_frame_id`，内容同样是 Twc |

`/slam/pose`、`/slam/path`、`/slam/map_points` 的 `header.frame_id` 都取自参数 `map_frame_id`，
`header.stamp` 取 `msg.header.stamp`；**若该时间戳为 0（sec 与 nanosec 都是 0）则退回节点时钟**
（`this->now()`），以免下游看到全 0 的时间戳。

### 3.2 参数（全部在构造函数里用 `declare_parameter` 带默认值声明）

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enable_pangolin` | bool | **false** | 是否让 ORB-SLAM3 开 Pangolin 可视化窗口 |
| `enable_opencv_window` | bool | **false** | 是否开 OpenCV 窗口（见 §7 已知限制 5） |
| `publish_pose` | bool | true | 是否发布 `/slam/pose` |
| `publish_path` | bool | true | 是否发布 `/slam/path` |
| `publish_points` | bool | true | 是否发布 `/slam/map_points` |
| `publish_tf` | bool | true | 是否广播 TF |
| `publish_status` | bool | true | 是否发布 `/slam/status` |
| `map_frame_id` | string | `"map"` | 位姿/点云/路径的父坐标系 |
| `camera_frame_id` | string | `"camera"` | TF 的子坐标系 |
| `max_path_length` | int | 2000 | `/slam/path` 最多保留多少个位姿；`0` 或负数表示不限制 |
| `point_publish_period_ms` | int | 1000 | 点云两次发布之间的最小间隔（毫秒）；`<= 0` 表示不限频（每帧都发） |

原有参数 `node_name_arg` / `voc_file_arg` / `settings_file_path_arg` 保持不变。

### 3.3 `/slam/status` 的 JSON 格式

一行 JSON（`std_msgs/String.data`），键固定如下：

```json
{"tracking_state":"OK","tracking_state_code":2,"n_map_points":1536,"image_scale":1.2,"fps":29.98}
```

* `tracking_state`(string) / `tracking_state_code`(int)：映射关系固定为
  `-1 SYSTEM_NOT_READY`、`0 NO_IMAGES_YET`、`1 NOT_INITIALIZED`、`2 OK`、`3 RECENTLY_LOST`、
  `4 LOST`、`5 OK_KLT`，其它值输出 `UNKNOWN`。
  代码里**只对整数做 switch**，不直接引用 `ORB_SLAM3::Tracking::eTrackingState` 枚举。
* `n_map_points`(int)：`GetAllMapPoints().size()`（为空时用 `GetTrackedMapPoints().size()`）。
* `image_scale`(float)：`GetImageScale()`。
* `fps`(float)：用相邻两帧图像时间戳之差估算，即 `1 / (stamp_n - stamp_{n-1})`；
  时间戳相同/回退时保持上一次的值。注意这是**图像时间戳的推算值**，不是 SLAM 的耗时。

---

## 4. 关键决策（接手者必读）

### 4.1 Twc 与 Tcw 的区别（最容易搞反的地方）

* `TrackMonocular()` 返回的是 **Tcw**：世界(地图)坐标系 → 相机坐标系，`X_cam = Tcw · X_world`。
* 发布的 **Twc = Tcw.inverse()**：相机在世界坐标系中的位姿，`X_world = Twc · X_cam`。
  * `/slam/pose`：`position = Twc.translation()`，`orientation = Twc.rotationMatrix()` 转成的四元数。
  * `/slam/path`：同上，逐帧累积。
  * TF `map → camera`：`TransformStamped` 的语义是「把点从 `child_frame_id` 变换到
    `header.frame_id`」，即 `^map T_camera = Twc`，所以**填的同样是 Twc**。

代码里用 `se3ToPoseMsg(const Sophus::SE3f& T)` 做转换，函数头的注释明确写了「调用方必须传 Twc」。

### 4.2 四元数顺序

`Sophus::SE3f::rotationMatrix()` 返回 `Eigen::Matrix3f`，用它构造 `Eigen::Quaternionf`。
Eigen 的访问器是 `q.x() / q.y() / q.z() / q.w()`，与 `geometry_msgs` 四元数的成员顺序
**x, y, z, w 完全一致**，所以直接一一赋值即可，不需要交换任何分量。
赋值前还调用 `quaternion.normalize()`，因为 RViz / TF 对非单位四元数很敏感。

### 4.3 为什么所有发布都写在 `Img_callback()` 里

**ORB-SLAM3 不是线程安全的。** `TrackMonocular()` / `GetTrackingState()` / `GetAllMapPoints()` /
`GetTrackedMapPoints()` / `GetImageScale()` 都会访问 SLAM 内部共享状态。
因此：

* 绝不新开线程、定时器或第二个回调去读 SLAM 对象；
* 所有发布都在跟踪完成之后**同步**执行；
* 需要降频的话题（点云）用参数 `point_publish_period_ms` + `std::chrono::steady_clock` 限频，
  而不是「换个线程降低频率」。

限频用 `steady_clock`（单调时钟）而不是 `rclcpp::Time`，是为了完全避开 ROS 时钟类型
（`RCL_ROS_TIME` / `RCL_SYSTEM_TIME`）的运算，防止运行期因时钟类型不匹配抛异常。

### 4.4 为什么默认关闭 Pangolin（`enable_pangolin` 默认 false）

`enablePangolinWindow = true` 时 ORB-SLAM3 会创建 Viewer 线程并初始化 Pangolin/OpenGL 上下文。
在没有显示器、没有 X11/EGL 的机器上（纯 SSH、CI、Docker、systemd 服务）会直接失败或崩溃，
典型报错是 `Pangolin X11: Unable to retrieve framebuffer options` 或 `GLX: Failed to create context`。
默认 `false` 才能保证「什么都不配也能把节点跑起来」。需要可视化时显式打开：

```bash
--ros-args -p enable_pangolin:=true
```

### 4.5 其余安全性与正确性修复

1. `pAgent` 在头文件里就初始化为 `nullptr`（原来是未初始化的野指针），
   析构函数与 `Img_callback()` 都先判空。
2. 收到图像但 `pAgent == nullptr`（driver 可能比握手先发图）时安全跳过，
   并且**警告限频 5 秒**，不会刷爆日志、不会崩。
3. `initializeVSLAM()` 的 settings 路径改为幂等：构造时把目录存进新成员 `settingsDirPath`
   （保证以 `/` 结尾），每次初始化都由 `settingsDirPath + configString + ".yaml"` 重新拼，
   **不再就地 `append` 成员**。重复握手因此不会拼出 `dir/name.yamlname.yaml`。
   注意 `settings_file_path_arg` 的语义是**目录**（与原实现一致），例如
   `.../orb_slam3/config/Monocular/`。
4. 重复收到 `experiment_settings` 时，先 `Shutdown()` 旧的 `pAgent` 再建新的，避免线程泄漏；
   同时清空上一轮的轨迹与统计量（`cameraPath.poses`、fps、限频状态）。
5. `TrackMonocular()` 之前多了一层防御：`cv_ptr` 判空 + `cv_ptr->image.empty()` 判空。
6. `GetAllMapPoints()` 返回空 vector（刚初始化/没建图）时退回 `GetTrackedMapPoints()`；
   遍历时对每个裸指针判空，遇到空指针该点写 `(0,0,0)` 并把 `is_dense` 置为 `false`。
7. 只发布需要的量：`GetAllMapPoints()` 会复制整个地图点指针数组、开销不小，
   因此 `/slam/status` 的 `n_map_points` 与 `/slam/map_points` **共用同一次查询结果**。

### 4.6 PointCloud2 的内存布局（手工设置，没走 `PointCloud2Modifier`）

```text
height      = 1
width       = N（点数）
point_step  = 12          （3 × sizeof(float)）
row_step    = 12 * N
is_bigendian= false       （小端）
is_dense    = true        （遇到空指针改 false）
fields[0]   = {name:"x", offset:0,  datatype:PointField::FLOAT32(7), count:1}
fields[1]   = {name:"y", offset:4,  datatype:PointField::FLOAT32(7), count:1}
fields[2]   = {name:"z", offset:8,  datatype:PointField::FLOAT32(7), count:1}
data        = 长度恰好 12 * N，每个点用 memcpy 写入 3 个 float
```

`N == 0` 时长度为 0 也是合法消息（发布空点云，RViz 不画东西）。

---

## 5. 构建

目标机：Ubuntu 22.04 + ROS2 Humble。设工作空间为 `~/ros2_ws`（按你的实际情况替换）。

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select ros2_orb_slam3
source install/setup.bash
```

说明：

* 本次改动只动了 C++ 节点与构建/依赖声明，**不涉及重新生成消息**，所以不需要 `--cmake-clean-cache`。
  若之前构建失败过、或换了依赖，建议干净重建：
  `rm -rf build/ros2_orb_slam3 install/ros2_orb_slam3 && colcon build --symlink-install --packages-select ros2_orb_slam3`
* 若 `orb_slam3/Thirdparty/DBoW2/lib/libDBoW2.so` 与 `orb_slam3/Thirdparty/g2o/lib/libg2o.so`
  还不存在，需要先按原 README 的方法编译这两个第三方库（本次改动没有碰它们的构建逻辑）。
* 运行节点（参数名沿用原来的）：

```bash
ros2 run ros2_orb_slam3 mono_node_cpp --ros-args \
  -p node_name_arg:=mono_node \
  -p voc_file_arg:=$HOME/ros2_ws/src/ros2_orb_slam3/orb_slam3/Vocabulary/ORBvoc.txt.bin \
  -p settings_file_path_arg:=$HOME/ros2_ws/src/ros2_orb_slam3/orb_slam3/config/Monocular/
```

  注意 `settings_file_path_arg` 要填**目录**（结尾带 `/`），文件名由 Python driver 通过
  `/mono_py_driver/experiment_settings` 发过来（例如发 `TUM1`，会拼成 `.../Monocular/TUM1.yaml`）。

---

## 6. 验证

前提：driver 已经完成握手（`/mono_py_driver/exp_settings_ack` 收到过 `ACK`）并开始发图，
否则 `/slam/*` 不会有数据（这是设计如此：`pAgent == nullptr` 时只打印限频警告）。

### 6.1 话题是否存在

```bash
ros2 topic list | grep slam
# 期望看到：
#   /slam/map_points
#   /slam/path
#   /slam/pose
#   /slam/status
```

### 6.2 位姿频率与内容

```bash
ros2 topic hz /slam/pose
# 期望约等于图像的发布频率（例如 20~30 Hz）

ros2 topic echo /slam/pose --once
# 检查 header.frame_id == "map"（或你设置的 map_frame_id）
# 检查 header.stamp 不是 0
# 检查 pose.orientation 四个分量不全为 0，且 x²+y²+z²+w² ≈ 1
```

### 6.3 状态 JSON

```bash
ros2 topic echo /slam/status --once
# data 字段是一段 JSON 字符串，形如：
# data: '{"tracking_state":"OK","tracking_state_code":2,"n_map_points":1536,"image_scale":1.2,"fps":29.98}'

# 只想看 data 本身：
ros2 topic echo /slam/status --once --field data
```

用 Python 解析校验（确认是合法 JSON 且字段齐全）：

```bash
ros2 topic echo /slam/status --once --field data --no-arr | tr -d "'" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d); assert {'tracking_state','tracking_state_code','n_map_points','image_scale','fps'} <= set(d)"
```

### 6.4 点云消息布局（验证手写的内存布局是否正确）

```bash
ros2 topic echo /slam/map_points --once --field point_step   # 期望 12
ros2 topic echo /slam/map_points --once --field height       # 期望 1
ros2 topic echo /slam/map_points --once --field width        # 期望 = 点数
ros2 topic echo /slam/map_points --once --field row_step     # 期望 = 12 * width
ros2 topic echo /slam/map_points --once --field fields
# 期望三个 PointField：x@0 / y@4 / z@8，datatype 都是 7 (FLOAT32)，count 都是 1

ros2 topic hz /slam/map_points
# 默认 point_publish_period_ms=1000，期望约 1 Hz
```

### 6.5 轨迹

```bash
ros2 topic hz /slam/path      # 与 /slam/pose 同频（每帧全量发布整条轨迹）
ros2 topic echo /slam/path --once --field poses   # 长度应随时间增长，直到 max_path_length=2000
```

### 6.6 TF

```bash
ros2 run tf2_ros tf2_echo map camera
# 期望持续输出 map -> camera 的平移/旋转，且与 /slam/pose 一致

ros2 run tf2_tools view_frames      # 生成 frames.pdf，确认 map -> camera 这条链存在
```

### 6.7 RViz2 里的步骤

```bash
rviz2
```

1. 左侧 **Global Options → Fixed Frame** 改成 `map`（必须与 `map_frame_id` 一致）。
2. **Add → By topic → /slam/path → Path**（或 Add → Path，然后在 Path → Topic 里选 `/slam/path`）。
   建议把 Color 设为固定颜色，Line Style 用 Lines。
3. **Add → By topic → /slam/map_points → PointCloud2**。
   在 PointCloud2 的属性里把 **Size (m)** 调成 `0.02` 左右、**Style** 选 `Points`，
   否则默认尺寸下点可能看不见。
4. **Add → TF**（可选）：能直观看到 `map` 与 `camera` 两个坐标系的相对关系。
5. 若 RViz 报 `Fixed Frame [map] does not exist`：说明还没握手成功、SLAM 还没发布 TF，
   或者 `map_frame_id` 被改成了别的名字（要把 Fixed Frame 改成同一个名字）。
6. 若想直接以相机为固定坐标系看：把 Fixed Frame 设为 `camera`，此时 `map` 会随相机移动。

---

## 7. 常见错误与排查

1. **Pangolin 打不开 / 节点启动即崩**
   症状：`Pangolin X11: Unable to retrieve framebuffer options`、
   `GLX: Failed to create context`、`terminate called after throwing an instance of 'std::runtime_error'`。
   原因：`enable_pangolin:=true` 但当前环境没有显示器/OpenGL 上下文。
   处理：保持默认 `false`；确实需要窗口时用有显示器的会话，或 `xvfb-run -a ros2 run ...`。

2. **`pkg-config` / 编译期找不到依赖**
   症状：`Could not find a package configuration file provided by "Pangolin"`、
   `Could not find "geometry_msgs"`、`Pangolin Found: FALSE`。
   处理：确认 ROS 环境已 source（`source /opt/ros/humble/setup.bash`），
   并把 Pangolin 的 `PangolinConfig.cmake` 所在目录加进来：
   `export Pangolin_DIR=/usr/local/lib/cmake/Pangolin`（路径按你的安装位置）。
   若报找不到 Eigen/Sophus，检查 `Eigen3_DIR` 与 `orb_slam3/Thirdparty/Sophus` 是否存在。

3. **运行期 `LD_LIBRARY_PATH` / 找不到 .so**
   症状：`error while loading shared libraries: libDBoW2.so: cannot open shared object file`、
   `libg2o.so`、`liborb_slam3_lib.so` 找不到。
   处理：先 `source install/setup.bash`（`install/` 下已经装了这些 .so），
   必要时显式补：
   `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ros2_ws/install/ros2_orb_slam3/lib`。
   也可以 `ldd $(ros2 pkg prefix ros2_orb_slam3)/lib/ros2_orb_slam3/mono_node_cpp | grep "not found"` 定位。

4. **TF 相关**
   * RViz 的 **Fixed Frame 必须等于 `map_frame_id`**（默认 `map`），否则看不到任何东西，
     并报 `Fixed Frame [map] does not exist`。
   * `/slam/pose` 的 `frame_id` 与 TF 的父坐标系都来自 `map_frame_id`，
     子坐标系来自 `camera_frame_id`（默认 `camera`）。两者不能写同一个名字。
   * 时间戳问题：如果 driver 发的图像 `header.stamp` 为 0，节点会用**节点时钟**，
     此时若下游开了 `/use_sim_time`（例如播 bag），会出现 `TF_OLD_DATA` / `extrapolation into the future`。
     建议让 driver 老老实实按图像时间打时间戳（例如 dataset 时间），全链路统一。
   * 只有握手成功之后才会广播 TF；`publish_tf:=false` 时当然也没有。

5. **`/slam/status` 里 `image_scale` 一直是 1，`tracking_state` 一直是 `NO_IMAGES_YET`**
   说明 SLAM 还没吃到有效图像：检查 `/mono_py_driver/timestep_msg` 是否在发、
   `/mono_py_driver/img_msg` 的编码能不能被 `cv_bridge` 转成灰度（ORB-SLAM3 单目需要单通道）。

6. **`/slam/map_points` 一直是空点云**
   正常：刚初始化、还没三角化出地图点时 `GetAllMapPoints()` 为空，
   此时会退回 `GetTrackedMapPoints()`，若仍为空就发布 `width: 0` 的空点云。
   等跟踪稳定（`tracking_state` 变成 `OK`）后点数会增长。

---

## 8. 应用 patch / 回滚

### 8.1 应用 patch

patch 里的路径是相对**包根目录**的（`a/CMakeLists.txt`、`a/src/common.cpp`、…），
所以要在包根目录下用 `-p1` 应用：

```bash
cd ~/ros2_ws/src/ros2_orb_slam3

# 1) 先检查能不能干净应用（--check 不会修改任何文件）
git apply --check patches/ros2_orb_slam3_slam_publishers.patch

# 2) 应用
git apply patches/ros2_orb_slam3_slam_publishers.patch

# 3) 确认改动的文件
git status --short
```

没有 git、或者不是 git 工作区时，用 GNU patch：

```bash
patch -p1 --dry-run < patches/ros2_orb_slam3_slam_publishers.patch
patch -p1 < patches/ros2_orb_slam3_slam_publishers.patch
```

说明：

* patch 是在 Windows 上生成的，但**内容是 LF 行尾**，在 Ubuntu 上应用没问题；
* 它只包含 4 个文件的**内容**改动，不含文件权限（mode）改动；
* 它**不包含**本次新增的两个文件（`SLAM_PUBLISHERS.md` 与 patch 文件自身），
  因为 `git diff` 不跟踪未加入索引的新文件。新文件请按常规方式保留/删除即可。

### 8.2 回滚

方式一：直接反向应用 patch（推荐，最精确）

```bash
cd ~/ros2_ws/src/ros2_orb_slam3
git apply -R --check patches/ros2_orb_slam3_slam_publishers.patch   # 先 dry-run
git apply -R patches/ros2_orb_slam3_slam_publishers.patch
# 或者 GNU patch：
patch -p1 -R < patches/ros2_orb_slam3_slam_publishers.patch
```

方式二：用 git 丢弃工作区改动

```bash
cd ~/ros2_ws/src/ros2_orb_slam3
git checkout -- include/ros2_orb_slam3/common.hpp src/common.cpp CMakeLists.txt package.xml scripts/mono_driver_node.py
```

（注意：`git checkout --` 会把这五个文件恢复到 **HEAD** 的状态。如果你在这五个文件上还有
别的未提交改动，会被一起丢掉，请先 `git stash` 或 `git diff > backup.patch`。）

方式三：把本次新增的两个文件也删掉（可选）

```bash
rm -f patches/ros2_orb_slam3_slam_publishers.patch SLAM_PUBLISHERS.md
```

回滚后记得重新构建：

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select ros2_orb_slam3
```

---

## 9. 已知限制 / 未验证的部分（务必先读）

本次改动是在 **Windows 上手工完成**的，环境里没有 ROS2 Humble，也没有 ORB-SLAM3 的编译依赖。
以下内容**没有经过编译或运行验证**：

1. **未编译、未运行**。所有「期望看到」的输出都是按 API 语义和消息定义推断的，
   实际运行可能需要微调。C++17 语法层面已经做过括号/注释/字符串配对的静态检查，但这不等于编译通过。
2. **`tf2_ros::TransformBroadcaster` 的构造方式**：按 Humble 官方文档
   （tf2_ros 0.25.x）构造函数接受节点引用，代码写成
   `tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);`，
   与 Humble 官方 tf2 tutorial 的写法一致。若目标机的 tf2_ros 版本较老（只有 `Node::SharedPtr`
   重载），需要改成 `std::make_shared<tf2_ros::TransformBroadcaster>(this->shared_from_this())`，
   **且必须放在构造函数之外**（构造函数里调 `shared_from_this()` 会抛 `std::bad_weak_ptr`）。
   **✅ 已在目标机验证（2026-09-10）**：本机 tf2_ros 0.25.20-2jammy.20260325 只有一个
   模板构造函数，内部走 `rclcpp::create_publisher(node, ...)` → `get_node_topics_interface(&node)`
   → `node->get_node_topics_interface()`，**全程不调用 `shared_from_this()`**，
   因此构造函数里 `make_unique<TransformBroadcaster>(*this)` 是安全的（见 §10.2）。
3. **`rclcpp::Time` → `builtin_interfaces::msg::Time` 的隐式转换**（`stamp = this->now();`）
   是 rclcpp 的公开 API。**✅ 已在目标机验证（2026-09-10）**：
   `rclcpp/time.hpp:84` 存在 `operator builtin_interfaces::msg::Time() const`。
4. **`sensor_msgs::msg::PointField::FLOAT32`** 按 Humble 的消息定义（`uint8 FLOAT32 = 7`）使用，
   也写了 `point_field.hpp` 的显式 include。**✅ 已在目标机验证（2026-09-10）**：
   `point_field__struct.hpp` 中 `FLOAT32 = 7u` 与代码假设一致。
5. **`enable_opencv_window` 目前只是被读取和打印，没有实际的 `cv::imshow` 调用**：
   原代码里的 `cv::imshow` / `cv::waitKey` 本来就是注释掉的，本次刻意没有启用，
   以免在无显示器时又引入一个崩溃点。该参数目前仅作为占位，留待后续按需实现。
6. **`/slam/path` 每帧全量发布整条轨迹**：默认 2000 个位姿、30 Hz 时大约 1 MB/s 量级，
   带宽敏感的场景请调小 `max_path_length`（或关掉 `publish_path`）。
7. **`/slam/status` 目前每帧发布一次**（只有 `publish_status` 开关，没有独立的限频参数）。
   话题频率高时可以用 `ros2 topic hz /slam/status` 观察，必要时再加参数。
8. **`GetAllMapPoints()` 返回的是裸指针**：理论上存在「地图线程同时删除某个 MapPoint」导致
   悬垂指针的极小概率窗口。代码已对 `nullptr` 判空，但无法完全消除该竞争——这与上游
   ORB-SLAM3 官方 ROS wrapper 的做法一致。如果对稳定性要求极高，可以把点云发布进一步降频
   （调大 `point_publish_period_ms`）。
9. **跟踪丢失（`RECENTLY_LOST` / `LOST`）时仍然发布位姿**：ORB-SLAM3 会返回「最后一次成功的位姿」，
   代码照常发布，由订阅方读 `/slam/status` 自行判断该位姿是否可信。如果上层希望丢失时**停止**
   发布，需要改 `Img_callback()` 里的 `publishPose` 分支。
10. **`fps` 是推算值**：用相邻图像时间戳之差算的。若 driver 不设置 `header.stamp`（为 0），
    节点会改用节点时钟，此时 `fps` 反映的是「节点收到图像的实际频率」，而不是数据集频率。
    另外若两帧时间戳相同甚至回退，`fps` 会保持上一次的值（不会出现负数或 inf）。

如果编译或运行中出现与本文档不符的地方，请以**目标机的实际报错**为准，并优先怀疑
上面第 2、3、4 条涉及的 API 细节。

---

## 10. 目标机适配与验证记录（2026-09-10）

### 10.1 路径适配（ros2_test → orb_slam3_ws）

上游把工作区名硬编码为 `ros2_test`，而目标机的工作区实际位于
`/home/hu/underwater_slam_project`。已在两处修正：

| 位置 | 原值 | 新值 | 影响 |
| --- | --- | --- | --- |
| `include/ros2_orb_slam3/common.hpp` 的 `packagePath` | `"ros2_test/src/ros2_orb_slam3/"` | `"orb_slam3_ws/src/ros2_orb_slam3/"` | `voc_file_arg` / `settings_file_path_arg` **未显式传入时**的回退路径（拼接 `$HOME` 后指向 `/home/hu/underwater_slam_project/...`） |
| `scripts/mono_driver_node.py` 的 `home_dir` | `~/ros2_test/src/ros2_orb_slam3` | `~/underwater_slam_project/src/ros2_orb_slam3` | EuRoC 数据集回放（`TEST_DATASET/sample_euroc_MH05`）时的数据集查找路径 |

注意：`underwater_slam_ws` 的 `slam_pipeline.launch.py` 总是显式传入 `voc_file` 与
`settings_dir`，真实相机主线**不依赖**这两个回退路径；回退路径只影响手动
`ros2 run ros2_orb_slam3 mono_node_cpp` 时不带参数的情况。若将来工作区位置再变，
改这两处即可（或启动时显式传 `voc_file_arg` / `settings_file_path_arg`）。

### 10.2 目标机 API 验证结论

在目标机（Ubuntu 22.04 + ROS2 Humble，`tf2_ros 0.25.20-2jammy.20260325`，
`rclcpp` 同源）上逐条核对了 §9 的第 2、3、4 条 API 风险，结论如下：

1. **TransformBroadcaster 构造安全**（§9.2）：模板构造函数
   `TransformBroadcaster(NodeT&& node, ...)` 内部只调用
   `rclcpp::create_publisher<tf2_msgs::msg::TFMessage>(node, "/tf", qos, options)`；
   对 `Node&` 左值，`rclcpp::node_interfaces::get_node_topics_interface` 取 `&node`
   后直接调 `get_node_topics_interface()`，**不经过 `shared_from_this()`**。
   因此构造函数里的 `std::make_unique<tf2_ros::TransformBroadcaster>(*this)`
   不会抛 `std::bad_weak_ptr`，维持现状不改。
2. **`rclcpp::Time` 隐式转换存在**（§9.3）：`rclcpp/rclcpp/time.hpp:84`
   `operator builtin_interfaces::msg::Time() const`。
3. **`PointField::FLOAT32 == 7`**（§9.4）：`sensor_msgs/msg/detail/point_field__struct.hpp`
   `static constexpr uint8_t FLOAT32 = 7u`，与手写 PointCloud2 布局一致。

仍待验证的只有 §9.1（**编译 + 运行**）：目标机上尚未执行
`colcon build --symlink-install --packages-select ros2_orb_slam3`。
编译若失败，优先检查 Pangolin/DBoW2/g2o 依赖，其次再回头怀疑上述已核对过的 API。
