# 给 AI 协作者的项目约定

本工作区是"基于单目 SLAM 的水下建图"项目。**动手前先按顺序读这两份**：

1. `src/underwater_perception/docs/AI_HANDOFF.md` —— 交接总纲（目录地图、接口、已验证/未验证、故障手册、已知陷阱）
2. `src/underwater_perception/docs/UNVERIFIED.md` —— 哪些结论**不允许**当成已验证的事实

## 硬性约定

1. **分层不许破坏**：`underwater_perception/core/` 是纯 Python（只依赖 numpy/opencv/pyyaml），
   **不许 import rclpy**；ROS 相关代码只能放在 `nodes/`，且只做参数/消息/发布。
   原因：开发机没有 ROS，core 是唯一能被真实验证的部分。
2. **改接口要同步四处**：代码、`config/*.yaml`、`docs/INTERFACES.md`、受影响的测试。
   消息字段只增不改、不改语义（`/detections` 是既有 YOLO 契约的超集）。
3. **参数键名唯一来源**是 `core/params.py`；YAML 键名写错会静默失效，
   `test/test_configs.py` 会静态拦下。
4. **提交前必须全绿**：
   ```bash
   cd src/underwater_perception
   PYTHONPATH=. python3 -m pytest test -q            # 99 项
   PYTHONPATH=. python3 -m underwater_perception.tools.selftest --out /tmp/st
   ```
   改了算法就**必须**补合成数据测试（`core/synthetic.py` 提供立方体/棋盘格渲染与真值）。
5. **不许编造结果**：合成数据、真实数据、未验证必须分清；引用数字要能指到
   `selftest_report.json` 或某条 pytest 的输出。
6. **文件编码**：UTF-8 无 BOM、LF 换行。Windows 上不要用 PowerShell 5.1 的
   `Set-Content`/`-replace` 改源码（实测会把中文写成乱码并吞换行）。
7. **位姿方向**：对外发布的是 **Twc（世界<-相机）**；ORB-SLAM3 内部是 `Tcw`，用前要 `.inverse()`。
8. **单目尺度**：SLAM 地图不是米。不要把"已知边长算出的米"直接当地图坐标。

## 常用入口

| 目的 | 命令 |
|---|---|
| 算法自测（无 ROS/无相机） | `python3 -m underwater_perception.tools.selftest --out ./selftest_out` |
| 单测 | `cd src/underwater_perception && PYTHONPATH=. python3 -m pytest test -q` |
| 检测调参（看掩码） | `python3 -m underwater_perception.tools.cube_detect_cli --input <图> --print-mask` |
| 标定 | `python3 -m underwater_perception.tools.calibrate_cli --images <目录> --out <输出>` |
| 无相机跑 ROS 全链路 | `ros2 launch underwater_perception full_demo.launch.py` |
| SLAM 链路 | `ros2 launch underwater_perception slam_pipeline.launch.py voc_file:=... settings_dir:=...` |

## 修改 C++（ORB-SLAM3 封装）时
* 只允许改动 `orb_slam3_ws/src/ros2_orb_slam3/` 下的 `include/ros2_orb_slam3/common.hpp`、
  `src/common.cpp`、`CMakeLists.txt`、`package.xml`（本次补丁范围）。
* 改完重新生成 patch：`git diff > patches/ros2_orb_slam3_slam_publishers.patch`，
  并同步更新该包的 `SLAM_PUBLISHERS.md`。
* **ORB-SLAM3 不是线程安全的**：所有 SLAM API 调用都要留在图像回调里，不要另开线程。
