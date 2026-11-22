"""数据集落盘工具（纯 Python，CLI 与 ROS 节点共用）。

放在 ``core/`` 而不是各写一份，是为了保证两件事一致：
* 文件名格式（可排序、不重名）；
* 中文/空格路径的写入方式（``cv2.imwrite`` 在部分平台会失败）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "ensure_dataset_dirs",
    "frame_filename",
    "imwrite_unicode",
    "write_data_yaml",
    "write_session_json",
]


def ensure_dataset_dirs(root: str | Path) -> Tuple[Path, Path]:
    """创建 ``images/`` 与 ``labels/``，返回两者路径。"""
    base = Path(root).expanduser()
    images = base / "images"
    labels = base / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return images, labels


def frame_filename(prefix: str, index: int, stamp_ns: int, image_format: str) -> str:
    """生成可排序、不重名的帧文件名。"""
    ext = image_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix) or "frame"
    return f"{safe_prefix}_{int(index):06d}_{int(stamp_ns)}.{ext}"


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """支持中文/空格路径的图像写入（``cv2.imwrite`` 在部分平台会失败）。"""
    target = Path(path)
    suffix = target.suffix if target.suffix else ".jpg"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        buffer.tofile(str(target))
        return True
    except OSError:
        return False


def write_data_yaml(root: str | Path, names: Sequence[str]) -> str:
    """写一个 YOLO 数据集模板（标注完成后再按 train/valid 拆分并改路径）。"""
    clean: List[str] = [str(name).strip() for name in names if str(name).strip()]
    if not clean:
        raise ValueError("names 不能为空")
    text = (
        "# 由 capture_dataset / dataset_capture 生成的模板；标注并拆分后请修改 train/val/path\n"
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(clean)}\n"
        f"names: {clean}\n"
    )
    target = Path(root).expanduser() / "data.yaml"
    target.write_text(text, encoding="utf-8")
    return str(target)


def write_session_json(root: str | Path, payload: Dict) -> str:
    """把采集会话信息写到 ``capture_session.json``（便于复现与写报告）。"""
    target = Path(root).expanduser() / "capture_session.json"
    data = dict(payload)
    data.setdefault("finished_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)
