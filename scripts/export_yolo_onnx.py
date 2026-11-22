#!/usr/bin/env python3
"""把训练好的 ``best.pt`` 导出为 ``models/best.onnx``，并做一次推理自检。

安全策略：

* 覆盖 ``models/best.onnx`` 前，旧文件自动备份到 ``models/backup/``；
* 同时把 ``best.pt`` 复制到 ``models/best.pt``（同样备份旧文件）；
* 用 onnxruntime 创建一次推理会话，确认输入/输出形状正常。

用法::

    python3 scripts/export_yolo_onnx.py \
      --weights /home/hu/underwater_slam_project/runs/yolo/rubbish_v1/weights/best.pt

    # 已复制到 models/best.pt 后
    python3 scripts/export_yolo_onnx.py --weights models/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = WORKSPACE / "models" / "best.pt"
DEFAULT_ONNX = WORKSPACE / "models" / "best.onnx"
DEFAULT_PT = WORKSPACE / "models" / "best.pt"


def _log(message: str) -> None:
    print(message, flush=True)


def _backup(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    shutil.copy2(path, target)
    _log(f"[备份] {path} -> {target}")


def _verify_onnx(path: Path) -> dict:
    """用 onnxruntime 打开模型，返回输入/输出信息。"""
    try:
        import onnxruntime as ort
    except ImportError:
        _log("[警告] 未安装 onnxruntime，跳过 ONNX 推理自检")
        return {}
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in session.get_inputs()
    ]
    outputs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in session.get_outputs()
    ]
    _log("ONNX 自检通过:")
    for item in inputs:
        _log(f"  input  {item['name']} {item['shape']} {item['type']}")
    for item in outputs:
        _log(f"  output {item['name']} {item['shape']} {item['type']}")
    return {"inputs": inputs, "outputs": outputs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 best.onnx 并自检")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="训练得到的 best.pt")
    parser.add_argument("--out", default=str(DEFAULT_ONNX), help="输出 ONNX 路径")
    parser.add_argument("--pt-out", default=str(DEFAULT_PT), help="同时复制 best.pt 到这里")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true", help="动态 batch/尺寸（默认静态）")
    parser.add_argument("--no-simplify", action="store_true", help="不调用 onnx-simplifier")
    args = parser.parse_args(argv)

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        _log(f"[错误] 找不到权重: {weights}")
        return 3

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        _log(f"[错误] 需要 ultralytics: {exc}")
        return 3

    target = Path(args.out).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # 关键：先备份旧目标，再导出。否则 Ultralytics 可能直接写同名文件，
    # 旧模型在新模型落地前就被覆盖了（真实踩过）。
    _backup(target)

    _log(f"加载权重: {weights}")
    names = None
    exported: Path | None = None
    export_kwargs = dict(
        format="onnx",
        imgsz=int(args.imgsz),
        opset=int(args.opset),
        dynamic=bool(args.dynamic),
        simplify=not args.no_simplify,
        batch=1,
    )
    # 在临时目录里用不同文件名导出，避免覆盖 target；也避免 SameFileError
    with tempfile.TemporaryDirectory(prefix="yolo_export_") as temp_dir:
        temp_weights = Path(temp_dir) / "export_source.pt"
        shutil.copy2(weights, temp_weights)
        model = YOLO(str(temp_weights))
        names = model.names
        _log(f"类别: {names}")
        try:
            exported = Path(model.export(**export_kwargs)).expanduser().resolve()
        except Exception as exc:
            if export_kwargs["simplify"]:
                _log(f"[警告] simplify 导出失败（{exc}），改为 simplify=False 重试")
                export_kwargs["simplify"] = False
                try:
                    exported = Path(model.export(**export_kwargs)).expanduser().resolve()
                except Exception as exc2:
                    _log(f"[错误] ONNX 导出失败: {exc2}")
                    return 4
            else:
                _log(f"[错误] ONNX 导出失败: {exc}")
                return 4

        if exported is None or not exported.is_file():
            _log(f"[错误] 导出结束但找不到文件: {exported}")
            return 4
        shutil.copy2(exported, target)
    _log(f"[完成] ONNX -> {target} ({target.stat().st_size / 1e6:.1f} MB)")

    pt_target = Path(args.pt_out).expanduser().resolve()
    if weights != pt_target:
        pt_target.parent.mkdir(parents=True, exist_ok=True)
        _backup(pt_target)
        shutil.copy2(weights, pt_target)
        _log(f"[完成] best.pt -> {pt_target} ({pt_target.stat().st_size / 1e6:.1f} MB)")

    onnx_info = _verify_onnx(target)

    metadata_path = target.parent / "best_metadata.json"
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata.update(
        {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "classes": names,
            "imgsz": int(args.imgsz),
            "opset": int(args.opset),
            "onnx": str(target),
            "pt": str(pt_target),
            "source_weights": str(weights),
            "source_weight_name": weights.name,
            "onnx_inputs": onnx_info.get("inputs", []),
            "onnx_outputs": onnx_info.get("outputs", []),
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[完成] 元数据 -> {metadata_path}")
    _log("")
    _log("下一步：把 config/yolo_real_camera.yaml 的 model_path 改成:")
    _log(f"  {pt_target}")
    _log("然后启动：ros2 launch underwater_perception yolo_real.launch.py \\")
    _log(f"  model_path:={pt_target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
