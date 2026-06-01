from pathlib import Path
from typing import Any, Dict, List


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        image_size: int = 320,
        device: str = "0",
        use_half: bool = False,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install it with: "
                "python3 -m pip install --user --no-deps ultralytics"
            ) from exc

        if not model_path:
            raise ValueError("model_path is empty. Set yolo_model_path.")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.use_half = use_half

        self.model_suffix = Path(model_path).suffix.lower()
        self.is_tensorrt_engine = self.model_suffix == ".engine"

        # task="detect" removes Ultralytics warning for TensorRT engines.
        self.model = YOLO(model_path, task="detect")

    def detect(self, frame) -> List[Dict[str, Any]]:
        predict_args = {
            "verbose": False,
            "imgsz": self.image_size,
            "conf": self.confidence_threshold,
        }

        # TensorRT engines already know their backend/device.
        # Passing device/half is mainly useful for .pt models.
        if not self.is_tensorrt_engine:
            if self.device:
                predict_args["device"] = self.device
            predict_args["half"] = self.use_half

        results = self.model(frame, **predict_args)

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        names = result.names
        detections = []

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = xyxy

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox_x": float(x1),
                    "bbox_y": float(y1),
                    "bbox_width": float(x2 - x1),
                    "bbox_height": float(y2 - y1),
                    "bbox_center_x": float((x1 + x2) / 2.0),
                    "bbox_center_y": float((y1 + y2) / 2.0),
                }
            )

        return detections