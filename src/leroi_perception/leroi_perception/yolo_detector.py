import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Depth filter bounds (mm)
MIN_DEPTH_MM = 150
MAX_DEPTH_MM = 5000

# Top-down map display
MAP_W = 600
MAP_H = 600
MAP_X_RANGE_CM = 200
MAP_Z_RANGE_CM = 300
ROBOT_X = MAP_W // 2
ROBOT_Y = MAP_H - 50

# Position smoothing
SMOOTHING_ALPHA = 0.20
MAX_POSITION_JUMP_CM = 80

# Experimental: when True, report the RAW DepthAI depth sampled at the detection
# center (no region median, no temporal smoothing) so raw stereo accuracy can be
# evaluated on the feed. Set back to False to restore median-over-box + smoothing.
RAW_DEPTH_MODE = False

# Piecewise depth calibration: corrected_z = scale*measured + offset, chosen by
# range (split near the 2 ft mark, in MEASURED-depth terms). Applied to the
# forward distance only -- lateral X stays on the unscaled depth, which reads
# accurately.
# Near range fit from tape data (16/20/24 in): a ~constant +8 cm over-read, so a
# fixed offset fits far better than a scale. Far range pending >2 ft points.
# The raw depth jumps from ~85 (30in true) to ~134 (36in true) as contamination
# kicks in, so the split lives in that gap. Near range is a flat offset; far range
# is a scale+offset fit to noisy/contaminated data (rough beyond ~30in).
DEPTH_BREAKPOINT_CM = 100.0         # measured depth (cm)
DEPTH_NEAR_SCALE = 1.0
DEPTH_NEAR_OFFSET_CM = -8.2         # fit from 16/20/24/30in: flat ~8.2cm over-read
DEPTH_FAR_SCALE = 0.562            # fit from 36/48/60in: true = 0.562*measured + 14.5
DEPTH_FAR_OFFSET_CM = 14.5


def _clamp_box(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def _get_region_depth(
    depth: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[int, int]]]:
    h, w = depth.shape
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 2 or box_h <= 2:
        return None, None

    px = (x1 + x2) // 2
    py = (y1 + y2) // 2

    regions = [
        (x1 + int(box_w * 0.30), y1 + int(box_h * 0.30), x2 - int(box_w * 0.30), y2 - int(box_h * 0.30), "middle 40%"),
        (x1 + int(box_w * 0.25), y1 + int(box_h * 0.25), x2 - int(box_w * 0.25), y2 - int(box_h * 0.25), "middle 50%"),
        (x1 + int(box_w * 0.15), y1 + int(box_h * 0.15), x2 - int(box_w * 0.15), y2 - int(box_h * 0.15), "middle 70%"),
        (max(0, px - 18), max(0, py - 18), min(w, px + 18), min(h, py + 18), "center crop"),
        (x1, y1, x2, y2, "whole box"),
    ]

    for rx1, ry1, rx2, ry2, region_name in regions:
        rx1, ry1, rx2, ry2 = _clamp_box(rx1, ry1, rx2, ry2, w, h)

        if rx2 <= rx1 or ry2 <= ry1:
            continue

        crop = depth[ry1:ry2, rx1:rx2]
        valid = crop[(crop > MIN_DEPTH_MM) & (crop < MAX_DEPTH_MM)]
        total = crop.size

        if total == 0 or len(valid) < 10 or len(valid) / total < 0.02:
            continue

        return {
            "z": float(np.median(valid)),
            "region": (rx1, ry1, rx2, ry2),
            "region_name": region_name,
            "valid_ratio": len(valid) / total,
            "valid_pixels": len(valid),
        }, (px, py)

    return None, (px, py)


def _center_raw_depth_mm(
    depth: np.ndarray, px: int, py: int
) -> Optional[float]:
    """Raw DepthAI depth (mm) at the detection center pixel, no region median.

    Returns the single center-pixel value when it is valid; only if that exact
    pixel is invalid does it fall back to the nearest valid in a tiny 5x5 window
    (so the readout is essentially the camera's own depth, not our averaging).
    """
    h, w = depth.shape
    px = max(0, min(int(px), w - 1))
    py = max(0, min(int(py), h - 1))

    z = float(depth[py, px])
    if MIN_DEPTH_MM < z < MAX_DEPTH_MM:
        return z

    x1 = max(0, px - 2)
    x2 = min(w, px + 3)
    y1 = max(0, py - 2)
    y2 = min(h, py + 3)
    crop = depth[y1:y2, x1:x2]
    valid = crop[(crop > MIN_DEPTH_MM) & (crop < MAX_DEPTH_MM)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def _corrected_depth_cm(z_cm: float) -> float:
    """Piecewise depth calibration (scale*z + offset), split near the 2 ft mark."""
    if z_cm <= DEPTH_BREAKPOINT_CM:
        return DEPTH_NEAR_SCALE * z_cm + DEPTH_NEAR_OFFSET_CM
    return DEPTH_FAR_SCALE * z_cm + DEPTH_FAR_OFFSET_CM


def _class_color(name: str) -> Tuple[int, int, int]:
    name = name.lower()
    if "blue" in name:
        return (255, 0, 0)
    if "red" in name:
        return (0, 0, 255)
    return (0, 255, 0)


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        image_size: int = 320,
        device: str = "0",
        use_half: bool = False,
        class_names: Optional[Dict[int, str]] = None,
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

        self.model = YOLO(model_path, task="detect")
        self.class_names_override = class_names

        self._smoothed_objects: Dict[str, Any] = {}

    def detect(self, frame) -> List[Dict[str, Any]]:
        predict_args = {
            "verbose": False,
            "imgsz": self.image_size,
            "conf": self.confidence_threshold,
        }

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

        names = self.class_names_override if self.class_names_override else result.names
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

    def detect_3d(
        self,
        frame,
        depth_frame: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> List[Dict[str, Any]]:
        """Run YOLO detection and enrich each result with 3D coordinates from stereo depth."""
        self._cleanup_old_smoothed_objects()

        h, w = frame.shape[:2]
        detections = self.detect(frame)

        for det in detections:
            x1 = int(det["bbox_x"])
            y1 = int(det["bbox_y"])
            x2 = int(x1 + det["bbox_width"])
            y2 = int(y1 + det["bbox_height"])

            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w, h)

            depth_info, center = _get_region_depth(depth_frame, x1, y1, x2, y2)

            if depth_info is None or center is None:
                det.update({"x_cm": None, "y_cm": None, "z_cm": None, "depth_region": None})
                continue

            px, py = center

            if RAW_DEPTH_MODE:
                # Raw DepthAI depth at the box center, no median over the region
                # and no temporal smoothing.
                Z = _center_raw_depth_mm(depth_frame, px, py)
                if Z is None:
                    det.update({"x_cm": None, "y_cm": None, "z_cm": None, "depth_region": None})
                    continue

                X = (px - cx) * Z / fx
                Y = (py - cy) * Z / fy
                det.update(
                    {
                        "x_cm": X / 10.0,
                        "y_cm": Y / 10.0,
                        "z_cm": Z / 10.0,
                        "raw_x_cm": X / 10.0,
                        "raw_z_cm": Z / 10.0,
                        "depth_region": (px, py, px, py),
                        "depth_region_name": "raw center",
                        "depth_valid_ratio": 1.0,
                    }
                )
                continue

            Z = depth_info["z"]
            X = (px - cx) * Z / fx
            Y = (py - cy) * Z / fy

            raw_x_cm = X / 10.0
            raw_y_cm = Y / 10.0
            # Apply the piecewise depth calibration to the forward distance.
            raw_z_cm = _corrected_depth_cm(Z / 10.0)

            smooth_x_cm, smooth_z_cm = self._smooth_position(
                det["class_name"], raw_x_cm, raw_z_cm, px, py
            )

            det.update(
                {
                    "x_cm": smooth_x_cm,
                    "y_cm": raw_y_cm,
                    "z_cm": smooth_z_cm,
                    "raw_x_cm": raw_x_cm,
                    "raw_z_cm": raw_z_cm,
                    "depth_region": depth_info["region"],
                    "depth_region_name": depth_info["region_name"],
                    "depth_valid_ratio": depth_info["valid_ratio"],
                }
            )

        return detections

    def _smooth_position(
        self, name: str, x_cm: float, z_cm: float, px: int, py: int
    ) -> Tuple[float, float]:
        grid_x = px // 80
        grid_y = py // 80
        key = f"{name}_{grid_x}_{grid_y}"

        if key not in self._smoothed_objects:
            self._smoothed_objects[key] = {
                "x_cm": x_cm,
                "z_cm": z_cm,
                "last_seen": time.time(),
            }
            return x_cm, z_cm

        old_x = self._smoothed_objects[key]["x_cm"]
        old_z = self._smoothed_objects[key]["z_cm"]

        jump = np.sqrt((x_cm - old_x) ** 2 + (z_cm - old_z) ** 2)
        if jump > MAX_POSITION_JUMP_CM:
            x_cm = old_x * 0.8 + x_cm * 0.2
            z_cm = old_z * 0.8 + z_cm * 0.2

        smooth_x = old_x * (1 - SMOOTHING_ALPHA) + x_cm * SMOOTHING_ALPHA
        smooth_z = old_z * (1 - SMOOTHING_ALPHA) + z_cm * SMOOTHING_ALPHA

        self._smoothed_objects[key]["x_cm"] = smooth_x
        self._smoothed_objects[key]["z_cm"] = smooth_z
        self._smoothed_objects[key]["last_seen"] = time.time()

        return smooth_x, smooth_z

    def _cleanup_old_smoothed_objects(self, max_age_s: float = 2.0) -> None:
        now = time.time()
        stale = [k for k, v in self._smoothed_objects.items() if now - v["last_seen"] > max_age_s]
        for k in stale:
            del self._smoothed_objects[k]

    @staticmethod
    def class_color(name: str) -> Tuple[int, int, int]:
        return _class_color(name)

    @staticmethod
    def draw_map(detections: List[Dict[str, Any]]) -> np.ndarray:
        import cv2

        map_img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        map_img[:] = (25, 25, 25)

        grid_color = (60, 60, 60)
        for x in range(0, MAP_W, 50):
            cv2.line(map_img, (x, 0), (x, MAP_H), grid_color, 1)
        for y in range(0, MAP_H, 50):
            cv2.line(map_img, (0, y), (MAP_W, y), grid_color, 1)

        cv2.line(map_img, (ROBOT_X, 0), (ROBOT_X, MAP_H), (90, 90, 90), 2)
        cv2.circle(map_img, (ROBOT_X, ROBOT_Y), 12, (255, 255, 255), -1)
        cv2.putText(map_img, "OAK", (ROBOT_X - 22, ROBOT_Y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.line(map_img, (ROBOT_X, ROBOT_Y), (ROBOT_X - 180, 80), (80, 80, 80), 2)
        cv2.line(map_img, (ROBOT_X, ROBOT_Y), (ROBOT_X + 180, 80), (80, 80, 80), 2)

        cv2.putText(map_img, "Top Down Map", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(map_img, "X = left/right, Z = forward", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

        for z_cm_tick in range(50, MAP_Z_RANGE_CM + 1, 50):
            y = int(ROBOT_Y - (z_cm_tick / MAP_Z_RANGE_CM) * (ROBOT_Y - 40))
            cv2.line(map_img, (ROBOT_X - 8, y), (ROBOT_X + 8, y), (180, 180, 180), 2)
            cv2.putText(map_img, f"{z_cm_tick}cm", (ROBOT_X + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        for det in detections:
            x_cm = det.get("x_cm")
            z_cm = det.get("z_cm")

            if x_cm is None or z_cm is None:
                continue
            if z_cm < 0 or z_cm > MAP_Z_RANGE_CM:
                continue
            if abs(x_cm) > MAP_X_RANGE_CM:
                continue

            name = det.get("class_name", "")
            color = _class_color(name)

            map_x = int(ROBOT_X + (x_cm / MAP_X_RANGE_CM) * (MAP_W / 2 - 40))
            map_y = int(ROBOT_Y - (z_cm / MAP_Z_RANGE_CM) * (ROBOT_Y - 40))

            cv2.circle(map_img, (map_x, map_y), 10, color, -1)
            cv2.circle(map_img, (map_x, map_y), 13, (255, 255, 255), 1)
            cv2.putText(map_img, f"{name} {z_cm:.0f}cm", (map_x + 12, map_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(map_img, f"X:{x_cm:.0f}", (map_x + 12, map_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return map_img
