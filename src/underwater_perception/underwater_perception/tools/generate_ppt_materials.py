"""答辩/PPT 素材收集脚本（不需要 ROS）。

把分散在各处的图片产物收集到一个目录里，方便直接插进 PPT 或结题报告：

* ``selftest_out/``：合成数据自测的检测图、标定角点图、去畸变样例。
* YOLO 训练目录（默认 ``~/runs/detect/*``）：训练曲线 results.png、混淆矩阵、
  PR 曲线、验证集可视化 val_batch*.jpg。
* 可选 ``--cube-detect-out``：cube_detect_cli 输出的 annotated_*.png / mask_*.png。

典型用法：

.. code-block:: bash

    # 默认收集：selftest_out + ~/runs/detect -> ~/ppt_materials
    python -m underwater_perception.tools.generate_ppt_materials

    # 指定工作区、训练目录与输出目录
    python -m underwater_perception.tools.generate_ppt_materials \
        --ws /home/hu/underwater_slam_project \
        --yolo-runs /home/hu/runs/detect \
        --cube-detect-out ./cube_demo_out \
        --out /home/hu/ppt_materials

输出：``--out`` 目录下带来源前缀的素材图（``selftest_*.png``、``yolo_<run>_*.png`` 等）。
退出码：0 完成（个别来源缺失只告警不失败）；1 参数错误。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SELFTEST_PATTERNS = [
    ("selftest_detect_", ["detect_*.png", "negative_*.png", "track_frame_*.png"]),
    ("selftest_calibration_", ["calibration_corners.png"]),
    ("selftest_undistorted_", ["cli_calibration/undistorted/*.jpg"]),
    ("selftest_cli_demo_", ["cli_demo/annotated_*.png"]),
]

YOLO_RUN_PATTERNS = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxPR_curve.png",
    "BoxR_curve.png",
    "labels.jpg",
    "val_batch*.jpg",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _collect(pattern_dir: Path, patterns, prefix: str, out: Path) -> list:
    """把 ``pattern_dir`` 下匹配 patterns 的文件复制到 out，文件名加前缀。"""
    copied = []
    for pattern in patterns:
        for src in sorted(pattern_dir.glob(pattern)):
            target = out / (prefix + src.name)
            shutil.copy2(src, target)
            copied.append(target.name)
    return copied


def _collect_selftest(ws: Path, out: Path) -> list:
    selftest_dir = ws / "selftest_out"
    if not selftest_dir.is_dir():
        _log(f"[警告] 找不到 selftest_out：{selftest_dir}（先跑 perception_selftest）")
        return []
    copied = []
    for prefix, patterns in SELFTEST_PATTERNS:
        copied += _collect(selftest_dir, patterns, prefix, out)
    return copied


def _collect_yolo_runs(yolo_runs: Path, out: Path) -> list:
    if not yolo_runs.is_dir():
        _log(f"[警告] 找不到 YOLO 训练目录：{yolo_runs}")
        return []
    copied = []
    run_dirs = [p for p in yolo_runs.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not run_dirs:
        _log(f"[警告] {yolo_runs} 下没有训练 run 目录")
        return copied
    for run_dir in sorted(run_dirs):
        # run 名做前缀时去掉可能出现的非法字符，保证文件名干净
        prefix = "yolo_" + run_dir.name + "_"
        copied += _collect(run_dir, YOLO_RUN_PATTERNS, prefix, out)
    return copied


def _collect_cube_detect(cube_out: Path, out: Path) -> list:
    if not cube_out or not Path(cube_out).is_dir():
        _log(f"[警告] 找不到 cube_detect 输出目录：{cube_out}（用 --cube-detect-out 指定）")
        return []
    copied = []
    copied += _collect(Path(cube_out), ["annotated_*.png"], "cube_", out)
    copied += _collect(Path(cube_out), ["mask_*.png"], "cube_mask_", out)
    return copied


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="收集自测/YOLO 训练/立方体识别的图片产物，供 PPT 与结题报告使用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ws", default="/home/hu/underwater_slam_project", help="underwater_slam_project 工作区根目录")
    parser.add_argument("--yolo-runs", default="/home/hu/runs/detect", help="YOLO 训练输出目录")
    parser.add_argument("--cube-detect-out", default="", help="cube_detect_cli 的输出目录（可选）")
    parser.add_argument("--out", default="/home/hu/ppt_materials", help="素材输出目录")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    copied += _collect_selftest(Path(args.ws), out)
    copied += _collect_yolo_runs(Path(args.yolo_runs), out)
    copied += _collect_cube_detect(args.cube_detect_out, out)

    _log(f"共收集 {len(copied)} 个素材文件到 {out}")
    for name in sorted(copied):
        _log(f"  - {name}")

    if copied:
        _log("\n下一步（照着做即可）：")
        _log("1) 打开输出目录挑选素材插入 PPT/报告")
        _log("2) 需要新素材时重跑对应工具（selftest / yolo detect / cube_detect_cli）后再执行本脚本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
