#!/usr/bin/env bash
# 录一段建图数据，用于回放/截图/结题材料。
#
# 用法:
#   bash scripts/record_mapping.sh                # 自动用时间戳命名
#   bash scripts/record_mapping.sh /path/to/bag   # 指定输出目录
#
# 按 Ctrl-C 结束录制。回放:
#   ros2 bag play <输出目录>
#   ros2 run underwater_perception slam_viewer
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"

OUT="${1:-${ROOT}/bags/mapping_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "${OUT}")"

echo "录制到: ${OUT}"
echo "Ctrl-C 结束"
exec ros2 bag record -o "${OUT}" \
  /image_enhanced \
  /yolo/annotated \
  /yolo/detections \
  /slam/pose \
  /slam/path \
  /slam/map_points \
  /slam/status \
  /semantic/landmarks \
  /semantic/markers
