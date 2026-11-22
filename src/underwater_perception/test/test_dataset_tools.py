"""数据集落盘、采集与体检工具的纯 Python 测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from underwater_perception.core.dataset_io import (
    ensure_dataset_dirs,
    frame_filename,
    imwrite_unicode,
    write_data_yaml,
)
from underwater_perception.tools import capture_dataset, yolo_dataset_check


def _write_image(path: Path, color: tuple = (20, 40, 60), size: tuple = (64, 48)) -> None:
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    assert imwrite_unicode(path, image)


def test_frame_filename_is_sortable_and_sanitized() -> None:
    assert frame_filename("frame", 3, 123456789, "jpeg") == "frame_000003_123456789.jpg"
    assert frame_filename("my clip/01", 0, 1, "png") == "my_clip_01_000000_1.png"


def test_dataset_io_writes_unicode_paths_and_yaml(tmp_path: Path) -> None:
    root = tmp_path / "水下 数据集"
    images, labels = ensure_dataset_dirs(root)
    target = images / "帧_000001.jpg"
    _write_image(target)
    assert target.is_file()

    yaml_path = write_data_yaml(root, ["cube", "marker"])
    text = Path(yaml_path).read_text(encoding="utf-8")
    assert "nc: 2" in text
    assert "cube" in text
    assert labels.is_dir()


def test_capture_dataset_headless_from_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()

    root = tmp_path / "raw"
    code = capture_dataset.main(
        [
            "--video", str(video_path),
            "--out", str(root),
            "--no-display",
            "--interval", "0.0",
            "--max", "3",
            "--format", "png",
            "--names", "cube,marker",
        ]
    )
    assert code == 0
    assert len(list((root / "images").glob("*.png"))) == 3
    assert (root / "data.yaml").is_file()
    assert (root / "capture_session.json").is_file()


def test_scan_dataset_reports_missing_invalid_duplicate_and_orphan(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    _write_image(root / "train" / "images" / "a.jpg")
    _write_image(root / "valid" / "images" / "c.jpg")  # 与 a.jpg 字节相同 -> 跨划分重复
    _write_image(root / "train" / "images" / "b.jpg", color=(90, 90, 90))
    _write_image(root / "valid" / "images" / "d.jpg", color=(120, 10, 10))

    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "valid" / "labels" / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "valid" / "labels" / "d.txt").write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    # b.jpg 故意不给标签
    (root / "train" / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = yolo_dataset_check.scan_dataset(root, ["cube"])
    assert report["n_images"] == 4
    assert report["n_labels"] == 3
    assert report["n_missing_labels"] == 1
    assert report["n_invalid_labels"] == 1
    assert report["n_cross_split_duplicates"] >= 1
    assert report["n_orphan_labels"] == 1
    assert report["class_instances"] == {"cube": 2}
    assert report["errors"]
