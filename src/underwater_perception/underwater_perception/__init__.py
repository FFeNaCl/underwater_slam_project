"""水下感知包（underwater_perception）。

分层约定：
    core/    纯算法，只依赖 numpy / opencv / pyyaml，可脱离 ROS 单测与 CLI 运行
    nodes/   ROS2 节点薄封装（rclpy），只做参数、消息转换与 core 调用
    tools/   命令行工具：离线标定、单图/目录检测、合成数据自测

线程/进程约定：core 内所有函数都是纯函数或无副作用对象，可在 ROS 回调里直接调用；
耗时操作（标定）建议放在节点的服务回调中并自行加锁。
"""

__version__ = "0.1.0"
