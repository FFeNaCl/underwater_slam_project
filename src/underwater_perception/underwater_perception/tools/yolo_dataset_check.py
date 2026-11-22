"""YOLO 数据集体检工具（纯 Python，无 ROS）。

训练前先跑一遍，能提前发现 90% 的"训练效果莫名其妙"：

* 图片与标签是否一一对应（漏标、孤立标签）；
* 标签行是否是合法的 ``class cx cy w h``（归一化到 0~1，宽高 >0）；
* 类别 id 是否超出 ``data.yaml`` 的 ``names`` 范围；
* 各类别实例数是否严重不均衡；
* **训练集/验证集是否有重复图片**（相邻帧随机拆分导致的泄漏，最常见）。

用法::

    python3 -m underwater_perception.tools.yolo_dataset_check --dataset ~/datasets/real_underwater
    python3 -m underwater_perception.tools.yolo_dataset_check --dataset . --write-data-yaml --names cube,marker

退出码：0 = 无结构错误；1 = 有错误；2 = 没找到图片。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

__all__ = ["IMAGE_SUFFIXES", "scan_dataset", "write_split_data_yaml", "main"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _load_names(root: Path, explicit: Sequence[str] = ()) -> List[str]:
    if explicit:
        return [str(name).strip() for name in explicit if str(name).strip()]
    data_path = root / "data.yaml"
    if not data_path.is_file():
        return []
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except OSError:
        return None
    if image is None:
        return None
    return int(image.shape[1]), int(image.shape[0])


def _split_of(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return "unlabeled"
    if parts and parts[0].lower() in SPLIT_NAMES:
        return SPLIT_NAMES[parts[0].lower()]
    return "unlabeled"


def _label_for_image(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.parent / "labels" / f"{image_path.stem}.txt"


def _parse_label(path: Path, n_classes: int) -> Tuple[List[Tuple[int, float, float, float, float]], List[str]]:
    """解析一个 YOLO 标签文件，返回合法行与错误描述。"""
    valid: List[Tuple[int, float, float, float, float]] = []
    errors: List[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return valid, [f"读取失败: {exc}"]
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"第 {line_no} 行字段不足 5 个: {raw!r}")
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            errors.append(f"第 {line_no} 行无法解析: {raw!r}")
            continue
        if class_id < 0 or (n_classes > 0 and class_id >= n_classes):
            errors.append(f"第 {line_no} 行 class_id={class_id} 超出 names 范围 0~{n_classes - 1}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            errors.append(f"第 {line_no} 行坐标未归一化到 0~1: {raw!r}")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"第 {line_no} 行宽或高为 0: {raw!r}")
            continue
        valid.append((class_id, cx, cy, width, height))
    return valid, errors


def scan_dataset(root: str | Path, names: Sequence[str] = ()) -> Dict:
    """扫描数据集并返回体检报告（纯字典，便于 JSON 输出与单测）。"""
    base = Path(root).expanduser().resolve()
    class_names = _load_names(base, names)
    report: Dict = {
        "root": str(base),
        "names": class_names,
        "n_classes": len(class_names),
        "splits": {},
        "n_images": 0,
        "n_labels": 0,
        "n_missing_labels": 0,
        "n_empty_labels": 0,
        "n_invalid_labels": 0,
        "n_orphan_labels": 0,
        "n_duplicate_images": 0,
        "n_cross_split_duplicates": 0,
        "class_instances": {},
        "errors": [],
        "warnings": [],
    }
    images = sorted(path for path in base.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        report["errors"].append("没有找到任何图片")
        return report

    image_stems: Dict[Path, Path] = {}
    hashes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    class_counts: Counter = Counter()
    split_counts: Counter = Counter()
    sizes: Counter = Counter()

    for image_path in images:
        split = _split_of(image_path, base)
        split_counts[split] += 1
        image_stems[image_path] = _label_for_image(image_path)
        size = _image_size(image_path)
        if size is None:
            report["errors"].append(f"图片无法读取: {image_path}")
        else:
            sizes[f"{size[0]}x{size[1]}"] += 1
        digest = hashlib.md5(image_path.read_bytes()).hexdigest()
        hashes[digest].append((split, str(image_path)))

    for image_path, label_path in image_stems.items():
        if not label_path.is_file():
            report["n_missing_labels"] += 1
            report["warnings"].append(f"缺少标签: {image_path}")
            continue
        report["n_labels"] += 1
        valid, errors = _parse_label(label_path, len(class_names))
        if not valid and not errors:
            report["n_empty_labels"] += 1
        for class_id, *_ in valid:
            class_counts[class_id] += 1
        for message in errors:
            report["errors"].append(f"{label_path}: {message}")
        report["n_invalid_labels"] += len(errors)

    referenced_labels = {path.resolve() for path in image_stems.values() if path.is_file()}
    for label_path in base.rglob("labels/*.txt"):
        if label_path.resolve() not in referenced_labels:
            report["n_orphan_labels"] += 1
            report["warnings"].append(f"孤立标签（没有对应图片）: {label_path}")

    for digest, occurrences in hashes.items():
        if len(occurrences) <= 1:
            continue
        report["n_duplicate_images"] += len(occurrences) - 1
        splits = {split for split, _path in occurrences}
        if len(splits) > 1:
            report["n_cross_split_duplicates"] += len(occurrences) - 1
            report["errors"].append(
                "跨数据集重复图片（训练/验证泄漏）: " + ", ".join(path for _split, path in occurrences)
            )

    report["n_images"] = len(images)
    report["splits"] = dict(split_counts)
    report["image_sizes"] = dict(sizes)
    report["class_instances"] = {
        (class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"): int(count)
        for class_id, count in sorted(class_counts.items())
    }
    if report["n_missing_labels"]:
        report["warnings"].append(
            f"{report['n_missing_labels']} 张图没有标签；若是负样本请保留空 txt，否则补标。"
        )
    if report["n_empty_labels"]:
        report["warnings"].append(
            f"{report['n_empty_labels']} 个空标签视为负样本（背景图），这是合法的；比例建议 <20%。"
        )
    if len(report["class_instances"]) > 1:
        counts = list(report["class_instances"].values())
        if max(counts) > 5 * max(min(counts), 1):
            report["warnings"].append("类别实例数相差超过 5 倍：建议补采少数类或使用类别权重。")
    if "valid" not in split_counts or "train" not in split_counts:
        report["warnings"].append("没有同时找到 train/ 与 valid/：训练前请按场次拆分，不要用同一批图做验证。")
    return report


def write_split_data_yaml(root: str | Path, names: Sequence[str]) -> str:
    """按 ``root/train/images`` 与 ``root/valid/images`` 写 data.yaml。"""
    clean = [str(name).strip() for name in names if str(name).strip()]
    if not clean:
        raise ValueError("生成 data.yaml 需要 --names（类别名，逗号分隔）")
    base = Path(root).expanduser()
    text = (
        "# 由 yolo_dataset_check.py 生成\n"
        f"path: {base}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(clean)}\n"
        f"names: {clean}\n"
    )
    target = base / "data.yaml"
    target.write_text(text, encoding="utf-8")
    return str(target)


def _print_report(report: Dict) -> None:
    print("=" * 74)
    print("YOLO 数据集体检报告")
    print("=" * 74)
    print(f"根目录      : {report['root']}")
    print(f"图片/标签   : {report['n_images']} / {report['n_labels']}")
    print(f"划分        : {report['splits']}")
    print(f"图像尺寸    : {report['image_sizes']}")
    print(f"类别实例数  : {report['class_instances']}")
    print(f"缺失标签    : {report['n_missing_labels']}（负样本请保留空 txt）")
    print(f"空标签      : {report['n_empty_labels']}（视为背景图）")
    print(f"非法标签行  : {report['n_invalid_labels']}")
    print(f"孤立标签    : {report['n_orphan_labels']}")
    print(f"重复图片    : {report['n_duplicate_images']}（跨划分 {report['n_cross_split_duplicates']}）")
    if report["errors"]:
        print("\n[错误]")
        for message in report["errors"][:30]:
            print("  -", message)
        if len(report["errors"]) > 30:
            print(f"  ... 另有 {len(report['errors']) - 30} 条")
    if report["warnings"]:
        print("\n[警告]")
        for message in report["warnings"][:30]:
            print("  -", message)
        if len(report["warnings"]) > 30:
            print(f"  ... 另有 {len(report['warnings']) - 30} 条")
    print("")


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="YOLO 数据集结构/标签/泄漏检查")
    parser.add_argument("--dataset", required=True, help="数据集根目录")
    parser.add_argument("--names", default="", help="类别名（逗号分隔）；默认读 data.yaml")
    parser.add_argument("--json", default="", help="把报告写到 JSON 文件")
    parser.add_argument("--write-data-yaml", action="store_true", help="按 train/valid 目录生成 data.yaml")
    args = parser.parse_args(argv)

    root = Path(args.dataset).expanduser()
    if not root.is_dir():
        print(f"[错误] 目录不存在: {root}", file=sys.stderr)
        return 2
    names = [part.strip() for part in args.names.split(",") if part.strip()]
    report = scan_dataset(root, names)
    _print_report(report)
    if args.json:
        Path(args.json).expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入: {args.json}")
    if args.write_data_yaml:
        try:
            path = write_split_data_yaml(root, names or report["names"])
            print(f"data.yaml 已写入: {path}")
        except ValueError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 1

    if report["n_images"] == 0:
        return 2
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
