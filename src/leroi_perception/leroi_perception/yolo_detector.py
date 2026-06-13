import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Depth filter bounds (mm) -- raw DepthAI depth frame is natively in millimeters.
MIN_DEPTH_MM = 150
MAX_DEPTH_MM = 5000

# Top-down map display (inches)
MAP_W = 600
MAP_H = 600
MAP_X_RANGE_IN = 78.74    # +/- lateral extent shown
MAP_Z_RANGE_IN = 118.11   # forward extent shown
ROBOT_X = MAP_W // 2
ROBOT_Y = MAP_H - 50

# Position smoothing
SMOOTHING_ALPHA = 0.20
MAX_POSITION_JUMP_IN = 31.5

# Experimental: when True, report the RAW DepthAI depth sampled at the detection
# center (no region median, no temporal smoothing) so raw stereo accuracy can be
# evaluated on the feed. Set back to False to restore median-over-box + smoothing.
RAW_DEPTH_MODE = False

# Piecewise depth calibration: corrected_z = scale*measured + offset, chosen by
# range (split near the 2 ft mark, in MEASURED-depth terms). Applied to both the
# forward distance and the lateral X (lateral is projected from the corrected
# depth so sideways map positions track the same calibration).
# Near range fit from tape data (16/20/24 in): a ~constant +3.23 in over-read, so
# a fixed offset fits far better than a scale. Far range pending >2 ft points.
# The raw depth jumps from ~33in (30in true) to ~53in (36in true) as
# contamination kicks in, so the split lives in that gap. Near range is a flat
# offset; far range is a scale+offset fit to noisy/contaminated data (rough
# beyond ~30 in). Values converted from the original metric fit (/2.54).
DEPTH_BREAKPOINT_IN = 39.37        # measured depth (in)
DEPTH_NEAR_SCALE = 1.0
DEPTH_NEAR_OFFSET_IN = -3.23       # flat over-read in the near range
DEPTH_FAR_SCALE = 0.562           # unitless; true = 0.562*measured + 5.71 in
DEPTH_FAR_OFFSET_IN = 5.71

# Millimeters per inch, for converting the raw depth frame to inches.
MM_PER_IN = 25.4


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


def _corrected_depth_in(z_in: float) -> float:
    """Piecewise depth calibration (scale*z + offset), split near the 2 ft mark."""
    if z_in <= DEPTH_BREAKPOINT_IN:
        return DEPTH_NEAR_SCALE * z_in + DEPTH_NEAR_OFFSET_IN
    return DEPTH_FAR_SCALE * z_in + DEPTH_FAR_OFFSET_IN


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
            # Only ultralytics *itself* being absent should suggest a pip
            # install. A failure deeper in the import chain — most commonly
            # torch failing to load a CUDA shared lib (e.g. libcudss.so.0 from
            # the pip nvidia-cu12 package, which is only discoverable via
            # LD_LIBRARY_PATH) — must surface its real cause. Reporting that as
            # "ultralytics not installed" previously sent us chasing a phantom
            # USB/camera-enumeration bug.
            if getattr(exc, "name", None) == "ultralytics":
                raise RuntimeError(
                    "ultralytics is not installed. Install it with: "
                    "python3 -m pip install --user --no-deps ultralytics"
                ) from exc
            raise RuntimeError(
                f"Failed to import ultralytics ({type(exc).__name__}: {exc}). "
                "This is a dependency-load failure, NOT a missing ultralytics "
                "package — typically a torch/CUDA library missing from "
                "LD_LIBRARY_PATH (e.g. libcudss.so.0 from the nvidia-cu12 "
                "package). Check that the CUDA lib dirs are on LD_LIBRARY_PATH."
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
        """Run YOLO detection and enrich each result with 3D coordinates (inches) from stereo depth."""
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
                det.update({"x_in": None, "y_in": None, "z_in": None, "depth_region": None})
                continue

            px, py = center

            if RAW_DEPTH_MODE:
                # Raw DepthAI depth at the box center, no median over the region
                # and no temporal smoothing. No calibration applied (raw eval mode).
                Z = _center_raw_depth_mm(depth_frame, px, py)
                if Z is None:
                    det.update({"x_in": None, "y_in": None, "z_in": None, "depth_region": None})
                    continue

                z_in = Z / MM_PER_IN
                x_in = (px - cx) * z_in / fx
                y_in = (py - cy) * z_in / fy
                det.update(
                    {
                        "x_in": x_in,
                        "y_in": y_in,
                        "z_in": z_in,
                        "raw_x_in": x_in,
                        "raw_z_in": z_in,
                        "depth_region": (px, py, px, py),
                        "depth_region_name": "raw center",
                        "depth_valid_ratio": 1.0,
                    }
                )
                continue

            # Apply the piecewise depth calibration to the forward distance, then
            # project BOTH lateral X and vertical Y from that corrected depth so
            # the sideways position tracks the same calibration.
            z_meas_in = depth_info["z"] / MM_PER_IN
            z_corr_in = _corrected_depth_in(z_meas_in)

            raw_x_in = (px - cx) * z_corr_in / fx
            raw_y_in = (py - cy) * z_corr_in / fy
            raw_z_in = z_corr_in

            smooth_x_in, smooth_z_in = self._smooth_position(
                det["class_name"], raw_x_in, raw_z_in, px, py
            )

            det.update(
                {
                    "x_in": smooth_x_in,
                    "y_in": raw_y_in,
                    "z_in": smooth_z_in,
                    "raw_x_in": raw_x_in,
                    "raw_z_in": raw_z_in,
                    "depth_region": depth_info["region"],
                    "depth_region_name": depth_info["region_name"],
                    "depth_valid_ratio": depth_info["valid_ratio"],
                }
            )

        return detections

    def _smooth_position(
        self, name: str, x_in: float, z_in: float, px: int, py: int
    ) -> Tuple[float, float]:
        grid_x = px // 80
        grid_y = py // 80
        key = f"{name}_{grid_x}_{grid_y}"

        if key not in self._smoothed_objects:
            self._smoothed_objects[key] = {
                "x_in": x_in,
                "z_in": z_in,
                "last_seen": time.time(),
            }
            return x_in, z_in

        old_x = self._smoothed_objects[key]["x_in"]
        old_z = self._smoothed_objects[key]["z_in"]

        jump = np.sqrt((x_in - old_x) ** 2 + (z_in - old_z) ** 2)
        if jump > MAX_POSITION_JUMP_IN:
            x_in = old_x * 0.8 + x_in * 0.2
            z_in = old_z * 0.8 + z_in * 0.2

        smooth_x = old_x * (1 - SMOOTHING_ALPHA) + x_in * SMOOTHING_ALPHA
        smooth_z = old_z * (1 - SMOOTHING_ALPHA) + z_in * SMOOTHING_ALPHA

        self._smoothed_objects[key]["x_in"] = smooth_x
        self._smoothed_objects[key]["z_in"] = smooth_z
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

        for z_in_tick in range(20, int(MAP_Z_RANGE_IN) + 1, 20):
            y = int(ROBOT_Y - (z_in_tick / MAP_Z_RANGE_IN) * (ROBOT_Y - 40))
            cv2.line(map_img, (ROBOT_X - 8, y), (ROBOT_X + 8, y), (180, 180, 180), 2)
            cv2.putText(map_img, f"{z_in_tick}in", (ROBOT_X + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        for det in detections:
            x_in = det.get("x_in")
            z_in = det.get("z_in")

            if x_in is None or z_in is None:
                continue
            if z_in < 0 or z_in > MAP_Z_RANGE_IN:
                continue
            if abs(x_in) > MAP_X_RANGE_IN:
                continue

            name = det.get("class_name", "")
            color = _class_color(name)

            map_x = int(ROBOT_X + (x_in / MAP_X_RANGE_IN) * (MAP_W / 2 - 40))
            map_y = int(ROBOT_Y - (z_in / MAP_Z_RANGE_IN) * (ROBOT_Y - 40))

            cv2.circle(map_img, (map_x, map_y), 10, color, -1)
            cv2.circle(map_img, (map_x, map_y), 13, (255, 255, 255), 1)
            cv2.putText(map_img, f"{name} {z_in:.0f}in", (map_x + 12, map_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(map_img, f"X:{x_in:.0f}", (map_x + 12, map_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return map_img
