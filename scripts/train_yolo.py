#!/usr/bin/env python3
"""训练 Roboflow YOLOv8 数据集，并输出 ``models/best.pt``。

针对本机 ``~/Downloads/rubbish.v1i.yolov8`` 这种**只有 train、没有 valid/test**
的导出做了两件事：

1. 先复制到一个独立目录，并按时间顺序切出验证集（默认最后 20%），
   不修改原始下载目录；避免“随机拆相邻帧”造成验证集泄漏。
2. 训练完成后把 ``weights/best.pt`` 复制到 ``models/best.pt``，
   旧文件自动备份到 ``models/backup/``。

用法::

    python3 scripts/train_yolo.py \
      --data /home/hu/Downloads/rubbish.v1i.yolov8 \
      --epochs 100 --imgsz 640 --batch 4 --device 0

    # 已经准备好的数据集（含 train/valid）也可以直接训练
    python3 scripts/train_yolo.py --data /path/to/data.yaml --skip-prepare

注意：13 张图远低于推荐量（每类 ≥200 实例），本次训练只能得到“能跑通”的
``best.pt``，精度不要当真；正式实验请继续采数据。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = WORKSPACE / "datasets" / "raw" / "rubbish.v1i.yolov8"
DEFAULT_OUT_DATASET = WORKSPACE / "datasets" / "rubbish_yolov8_prepared"
DEFAULT_PROJECT = WORKSPACE / "runs" / "yolo"
DEFAULT_TARGET_PT = WORKSPACE / "models" / "best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_root(path: str | Path) -> Path:
    """支持传数据集目录或 data.yaml。"""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"--data 应指向目录或 data.yaml，当前是文件: {root}")
        root = root.parent
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"数据集缺少 train/ 目录: {root}")
    return root


def _load_names(root: Path) -> List[str]:
    data_path = root / "data.yaml"
    if data_path.is_file():
        data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        names = data.get("names", ["rubbish"])
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return ["rubbish"]


def _split_dir(root: Path, split: str) -> Tuple[Path, Path]:
    base = root / split
    images = base / "images"
    labels = base / "labels"
    if not images.is_dir():
        # 兼容少数导出的 images/train 结构
        alternative = root / "images" / split
        if alternative.is_dir():
            images = alternative
            labels = root / "labels" / split
    return images, labels


def _images(root: Path, split: str) -> List[Path]:
    images_dir, _ = _split_dir(root, split)
    if not images_dir.is_dir():
        return []
    return sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def _copy_pair(image: Path, label_dir: Path, out_split: Path) -> None:
    out_images = out_split / "images"
    out_labels = out_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, out_images / image.name)
    source_label = label_dir / f"{image.stem}.txt"
    target_label = out_labels / f"{image.stem}.txt"
    if source_label.is_file():
        shutil.copy2(source_label, target_label)
    else:
        # 没有标签的图按负样本处理，写空 txt，避免 Ultralytics 反复警告
        target_label.write_text("", encoding="utf-8")


def prepare_dataset(
    source: str | Path,
    output: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "time",
    overwrite: bool = False,
) -> Tuple[Path, List[str]]:
    """把源数据集整理成 ``train/valid[/test]`` 并写 data.yaml。

    Returns:
        ``(data_yaml_path, names)``
    """
    src = _resolve_root(source)
    out = Path(output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            existing = out / "data.yaml"
            if existing.is_file():
                _log(f"[提示] 复用已存在的数据集: {out}")
                return existing, _load_names(out)
            raise FileExistsError(f"输出目录非空: {out}；加 --overwrite 或换 --out-dataset")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    names = _load_names(src)
    train_images = _images(src, "train")
    if not train_images:
        raise FileNotFoundError(f"没有找到训练图片: {src / 'train' / 'images'}")

    _, train_labels_dir = _split_dir(src, "train")
    valid_images = _images(src, "valid") or _images(src, "val")
    test_images = _images(src, "test")
    _, test_labels_dir = _split_dir(src, "test")

    if valid_images:
        _log(f"源数据集已有 valid: {len(valid_images)} 张，直接使用")
        train_part = train_images
        valid_part = valid_images
        _, valid_labels_dir = _split_dir(src, "valid")
        if not valid_labels_dir.is_dir():
            _, valid_labels_dir = _split_dir(src, "val")
    else:
        # 按文件名/时间排序后切最后一段，避免随机拆相邻帧
        ordered = sorted(train_images, key=lambda p: p.name)
        if split_mode == "random":
            random.Random(seed).shuffle(ordered)
        n_val = max(1, int(round(len(ordered) * float(val_ratio))))
        n_val = min(n_val, max(1, len(ordered) - 1))
        valid_part = ordered[-n_val:]
        train_part = ordered[:-n_val]
        # 关键：验证集是从 train 切的，标签也必须回到 train/labels 找，
        # 否则会把所有 valid 标签写成空文件（真实踩过）。
        valid_labels_dir = train_labels_dir
        _log(
            f"源数据集没有 valid：按 {'时间' if split_mode == 'time' else '随机'} 切分 "
            f"{len(train_part)} train / {len(valid_part)} valid"
        )

    for image in train_part:
        _copy_pair(image, train_labels_dir, out / "train")
    for image in valid_part:
        _copy_pair(image, valid_labels_dir, out / "valid")
    if test_images:
        for image in test_images:
            _copy_pair(image, test_labels_dir, out / "test")

    data = {
        "path": str(out),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": names,
    }
    if test_images:
        data["test"] = "test/images"
    data_path = out / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    _log(f"已准备数据集: {out}")
    _log(f"  train={len(train_part)}  valid={len(valid_part)}  test={len(test_images)}  names={names}")
    if len(train_part) < 50:
        _log(
            "[警告] 训练图片少于 50 张：模型只能用于流程验证，mAP 和真实检出率不能作为结论。"
        )
    return data_path, names


def _backup(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    shutil.copy2(path, target)
    _log(f"[备份] {path} -> {target}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 并复制 best.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="数据集目录或 data.yaml")
    parser.add_argument("--out-dataset", default=str(DEFAULT_OUT_DATASET), help="整理后的数据集目录")
    parser.add_argument("--skip-prepare", action="store_true", help="数据已准备好，直接训练")
    parser.add_argument("--prepare-only", action="store_true", help="只整理数据集，不训练")
    parser.add_argument("--overwrite", action="store_true", help="重建整理后的数据集目录")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--model", default="yolov8n.pt", help="基础权重，例如 yolov8n.pt / yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="", help='""=自动，0=第一块 GPU，cpu=CPU')
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="rubbish_v1")
    parser.add_argument("--copy-to", default=str(DEFAULT_TARGET_PT), help="训练后权重复制到哪里")
    parser.add_argument(
        "--copy-last",
        action="store_true",
        help="复制 last.pt 而不是 best.pt。验证集极小时 best.pt 可能选到退化模型，推荐开启",
    )
    args = parser.parse_args(argv)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        _log(f"[错误] 需要 ultralytics/torch: {exc}")
        return 3

    if args.skip_prepare:
        root = _resolve_root(args.data)
        data_path = root / "data.yaml"
        if not data_path.is_file():
            _log(f"[错误] 找不到 {data_path}")
            return 3
        names = _load_names(root)
    else:
        try:
            data_path, names = prepare_dataset(
                args.data,
                args.out_dataset,
                val_ratio=args.val_ratio,
                seed=args.seed,
                split_mode=args.split_mode,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            _log(f"[错误] 数据集准备失败: {exc}")
            return 3
        if args.prepare_only:
            _log(f"仅整理数据集：{data_path}")
            return 0

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    _log(
        f"开始训练: model={args.model} data={data_path} imgsz={args.imgsz} "
        f"epochs={args.epochs} batch={args.batch} device={device}"
    )
    if device == "cpu":
        _log("[警告] 未检测到可用 CUDA，将使用 CPU（会很慢）。")

    model = YOLO(args.model)
    try:
        model.train(
            data=str(data_path),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            device=device,
            project=str(Path(args.project).expanduser()),
            name=str(args.name),
            exist_ok=True,
            seed=int(args.seed),
            patience=int(args.patience),
            workers=int(args.workers),
            plots=True,
            verbose=True,
            # 小数据集增强：让 13 张图不至于完全记住原图
            degrees=10.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mosaic=0.5,
            close_mosaic=10,
        )
    except Exception as exc:
        _log(f"[错误] 训练失败: {exc}")
        return 4

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / args.name))
    weight_name = "last.pt" if args.copy_last else "best.pt"
    source_weight = save_dir / "weights" / weight_name
    if not source_weight.is_file():
        _log(f"[错误] 训练结束但找不到 {weight_name}: {source_weight}")
        return 5

    target = Path(args.copy_to).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _backup(target)
    shutil.copy2(source_weight, target)
    _log(f"[完成] {weight_name} -> {target}")

    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": str(data_path),
        "classes": names,
        "model": args.model,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": device,
        "train_dir": str(save_dir),
        "copied_weight": weight_name,
        "best_pt": str(target),
        "source_best_pt": str(source_weight),
        "note": "训练集很小，指标仅供流程验证；正式实验请补充真实水下数据。",
    }
    meta_path = target.parent / "best_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[完成] 元数据 -> {meta_path}")
    _log(f"训练结果目录: {save_dir}")
    _log("下一步导出 ONNX：")
    _log(f"  python3 scripts/export_yolo_onnx.py --weights {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
