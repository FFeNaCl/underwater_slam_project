from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .image_utils import DEFAULT_CLASS_NAMES, bgr_to_image_msg, draw_detections, image_msg_to_bgr, load_yolo_names


class YoloDetector(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detector")

        self.input_topic = str(self.declare_parameter("input_topic", "/image_enhanced").value)
        self.annotated_topic = str(self.declare_parameter("annotated_topic", "/yolo/annotated").value)
        self.detections_topic = str(self.declare_parameter("detections_topic", "/yolo/detections").value)
        self.label_topic = str(self.declare_parameter("label_topic", "/dataset/labels").value)
        self.dataset_root_value = str(self.declare_parameter("dataset_root", "").value)
        self.dataset_root = Path(self.dataset_root_value) if self.dataset_root_value else None
        self.model_path = str(self.declare_parameter("model_path", "").value)
        self.confidence_threshold = float(self.declare_parameter("confidence_threshold", 0.25).value)
        self.allow_label_replay = bool(self.declare_parameter("allow_label_replay", True).value)
        self.require_model = bool(self.declare_parameter("require_model", False).value)
        self.imgsz = int(self.declare_parameter("imgsz", 640).value)
        self.device = str(self.declare_parameter("device", "").value)
        self.half = bool(self.declare_parameter("half", False).value)
        self.max_det = int(self.declare_parameter("max_det", 100).value)
        self.frame_title = str(self.declare_parameter("frame_title", "Underwater YOLO detection").value)

        self.names = load_yolo_names(self.dataset_root) if self.dataset_root is not None else []
        self.model = self.try_load_model(self.model_path)
        self.names = self.resolve_class_names()
        self.latest_label_payload: Optional[Dict] = None

        self.image_sub = self.create_subscription(Image, self.input_topic, self.on_image, 10)
        self.label_sub = None
        if self.label_topic:
            self.label_sub = self.create_subscription(String, self.label_topic, self.on_label, 10)
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)

        if self.model is not None:
            mode = "ultralytics"
        elif self.require_model:
            mode = "model-required (waiting for model_path)"
        else:
            mode = "label replay"
        self.get_logger().info(
            f"YOLO detector started in {mode} mode: {self.input_topic} -> {self.annotated_topic}"
        )

    def resolve_class_names(self) -> List[str]:
        if self.names:
            return self.names
        model_names = getattr(self.model, "names", None)
        if isinstance(model_names, dict):
            return [str(model_names[k]) for k in sorted(model_names)]
        if isinstance(model_names, list):
            return [str(name) for name in model_names]
        return list(DEFAULT_CLASS_NAMES)

    def try_load_model(self, model_path: str):
        if not model_path:
            if self.require_model:
                self.get_logger().error(
                    "model_path is empty (require_model=true); set it to the trained best.pt path"
                )
            else:
                self.get_logger().warning("model_path is empty; using label replay when available")
            return None

        path = Path(model_path).expanduser()
        if not path.exists():
            message = f"model_path does not exist: {path}"
            if self.require_model:
                self.get_logger().error(message + " (require_model=true: 不会做任何回放)")
            else:
                self.get_logger().warning(message + "; using label replay when available")
            return None

        try:
            from ultralytics import YOLO

            model = YOLO(str(path), task="detect")
            self.get_logger().info(
                f"loaded YOLO model: {path} (imgsz={self.imgsz}, device={self.device or 'auto'}, "
                f"half={self.half}, max_det={self.max_det})"
            )
            return model
        except Exception as exc:
            self.get_logger().error(f"failed to load YOLO model {path}: {exc}")
            return None

    def on_label(self, msg: String) -> None:
        try:
            self.latest_label_payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("received invalid dataset label json")

    def infer_with_model(self, image) -> List[Dict]:
        if self.model is None:
            return []

        predict_kwargs = {
            "conf": self.confidence_threshold,
            "imgsz": int(self.imgsz),
            "max_det": int(self.max_det),
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device
        if self.half:
            predict_kwargs["half"] = True
        results = self.model.predict(image, **predict_kwargs)
        detections: List[Dict] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = [int(round(v)) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.names[class_id] if 0 <= class_id < len(self.names) else str(class_id),
                    "confidence": confidence,
                    "xyxy": xyxy,
                }
            )
        return detections

    def replay_labels(self) -> List[Dict]:
        if not self.allow_label_replay or not self.latest_label_payload:
            return []
        return list(self.latest_label_payload.get("detections", []))

    def on_image(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().error(f"failed to convert image: {exc}")
            return

        model_unavailable = self.model is None and self.require_model
        if self.model is not None:
            detections = self.infer_with_model(image)
            source = "model"
            title = self.frame_title
        elif model_unavailable:
            # 真实相机配置里 require_model=true：模型没填时必须在画面上明确提示，
            # 而不是发布一帧“看起来正常但没有任何框”的空结果。
            detections = []
            source = "model_unavailable"
            title = "MODEL NOT LOADED - set model_path to best.pt"
            self.get_logger().error(
                f"model_path 为空或文件不存在（{self.model_path!r}）：请填入训练好的 best.pt 路径",
                throttle_duration_sec=10.0,
            )
        else:
            detections = self.replay_labels()
            source = "label_replay"
            title = self.frame_title

        annotated = draw_detections(image, detections, title)
        self.annotated_pub.publish(bgr_to_image_msg(annotated, msg.header.stamp, msg.header.frame_id))

        out = String()
        out.data = json.dumps(
            {
                "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
                "frame_id": msg.header.frame_id,
                "source": source,
                "model_loaded": self.model is not None,
                "count": len(detections),
                "detections": detections,
            },
            ensure_ascii=True,
        )
        self.detections_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
