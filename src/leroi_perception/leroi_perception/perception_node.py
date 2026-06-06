import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from leroi_perception.camera_sources import create_camera_source
from leroi_perception.preview_streamer import MjpegPreviewStreamer
from leroi_perception.yolo_detector import YoloDetector

try:
    from leroi_interfaces.msg import DetectedObject, DetectionArray
    _INTERFACES_AVAILABLE = True
except ImportError:
    _INTERFACES_AVAILABLE = False


class PerceptionNode(Node):
    """
    Main perception node.

    Responsibilities:
      - Read camera frames from OAK / USB camera.
      - Run YOLO.
      - If OAK depth is enabled, convert YOLO detections to camera-relative X/Z.
      - Subscribe to /v5/odom.
      - Project detections into field coordinates.
      - Draw camera preview and top-down field map on port 8080.

    Coordinate convention for map:
      - Field is 365.76 cm x 365.76 cm.
      - (0, 0) is bottom-left of the displayed field map.
      - +X is right.
      - +Y is up/forward.
      - robot heading 0 deg means facing +Y.
    """

    def __init__(self):
        super().__init__("perception_node")

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------

        # Camera parameters
        self.declare_parameter("camera_name", "oak_front")
        self.declare_parameter("camera_type", "oak")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("oak_mxid", "")
        self.declare_parameter("oak_with_depth", False)

        # Preview/status parameters
        self.declare_parameter("preview_rate_hz", 30.0)
        self.declare_parameter("preview_stream_fps", 15.0)
        self.declare_parameter("status_rate_hz", 2.0)

        # YOLO parameters
        self.declare_parameter("enable_yolo", False)
        self.declare_parameter("yolo_rate_hz", 2.0)
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_confidence", 0.35)
        self.declare_parameter("yolo_image_size", 320)
        self.declare_parameter("yolo_device", "cpu")
        self.declare_parameter("yolo_use_half", False)
        self.declare_parameter("yolo_class_names", "")

        # Web preview parameters
        self.declare_parameter("enable_preview_stream", True)
        self.declare_parameter("preview_port", 8080)
        self.declare_parameter("preview_jpeg_quality", 60)
        self.declare_parameter("draw_detections", True)

        # Map parameters
        self.declare_parameter("enable_map_stream", True)
        self.declare_parameter("map_width_px", 700)
        self.declare_parameter("map_height_px", 700)

        # 12 ft x 12 ft VEX field = 365.76 cm x 365.76 cm
        self.declare_parameter("field_min_x_cm", 0.0)
        self.declare_parameter("field_max_x_cm", 365.76)
        self.declare_parameter("field_min_y_cm", 0.0)
        self.declare_parameter("field_max_y_cm", 365.76)

        self.declare_parameter(
            "field_map_image_path",
            "~/leroiai-jetsonWorkspace/field_map.png",
        )
        self.declare_parameter("field_map_rotation", "cw")  # none, cw, ccw, 180
        self.declare_parameter("field_map_alpha", 0.72)
        self.declare_parameter("draw_field_grid", True)

        self.declare_parameter("robot_trail_max_points", 500)
        self.declare_parameter("object_history_sec", 3.0)

        # Since decision now publishes approved path/action commands to /robot/command
        self.declare_parameter("target_path_topic", "/robot/command")
        self.declare_parameter("action_ack_topic", "/robot/action_ack")

        # Robot/camera transform.
        # x = camera offset to robot right, y = camera offset forward from robot center.
        self.declare_parameter("camera_offset_x_cm", 0.0)
        self.declare_parameter("camera_offset_y_cm", 15.0)
        self.declare_parameter("camera_offset_z_cm", 25.0)

        # Goal view stays disabled for now.
        self.declare_parameter("enable_goal_view", False)

        # ---------------------------------------------------------------------
        # Read parameters
        # ---------------------------------------------------------------------

        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_type = str(self.get_parameter("camera_type").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.oak_mxid = str(self.get_parameter("oak_mxid").value)
        self.oak_with_depth = bool(self.get_parameter("oak_with_depth").value)

        self.preview_rate_hz = float(self.get_parameter("preview_rate_hz").value)
        self.preview_stream_fps = float(self.get_parameter("preview_stream_fps").value)
        self.status_rate_hz = float(self.get_parameter("status_rate_hz").value)

        self.enable_yolo = bool(self.get_parameter("enable_yolo").value)
        self.yolo_rate_hz = float(self.get_parameter("yolo_rate_hz").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_confidence = float(self.get_parameter("yolo_confidence").value)
        self.yolo_image_size = int(self.get_parameter("yolo_image_size").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_use_half = bool(self.get_parameter("yolo_use_half").value)

        names_raw = str(self.get_parameter("yolo_class_names").value).strip()
        self.yolo_class_names = None

        if names_raw:
            try:
                parsed = json.loads(names_raw)
                self.yolo_class_names = {int(k): str(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, ValueError):
                self.get_logger().warning(
                    "yolo_class_names is not valid JSON. Ignoring."
                )

        self.enable_preview_stream = bool(
            self.get_parameter("enable_preview_stream").value
        )
        self.preview_port = int(self.get_parameter("preview_port").value)
        self.preview_jpeg_quality = int(
            self.get_parameter("preview_jpeg_quality").value
        )
        self.draw_detections = bool(self.get_parameter("draw_detections").value)

        self.enable_map_stream = bool(self.get_parameter("enable_map_stream").value)
        self.map_width_px = int(self.get_parameter("map_width_px").value)
        self.map_height_px = int(self.get_parameter("map_height_px").value)

        self.field_min_x_cm = float(self.get_parameter("field_min_x_cm").value)
        self.field_max_x_cm = float(self.get_parameter("field_max_x_cm").value)
        self.field_min_y_cm = float(self.get_parameter("field_min_y_cm").value)
        self.field_max_y_cm = float(self.get_parameter("field_max_y_cm").value)

        self.field_map_image_path = os.path.expanduser(
            str(self.get_parameter("field_map_image_path").value)
        )
        self.field_map_rotation = (
            str(self.get_parameter("field_map_rotation").value).lower().strip()
        )
        self.field_map_alpha = float(self.get_parameter("field_map_alpha").value)
        self.draw_field_grid = bool(self.get_parameter("draw_field_grid").value)

        self.robot_trail_max_points = int(
            self.get_parameter("robot_trail_max_points").value
        )
        self.object_history_sec = float(self.get_parameter("object_history_sec").value)

        self.target_path_topic = str(self.get_parameter("target_path_topic").value)
        self.action_ack_topic = str(self.get_parameter("action_ack_topic").value)

        self.camera_offset_x_cm = float(
            self.get_parameter("camera_offset_x_cm").value
        )
        self.camera_offset_y_cm = float(
            self.get_parameter("camera_offset_y_cm").value
        )
        self.camera_offset_z_cm = float(
            self.get_parameter("camera_offset_z_cm").value
        )

        self.enable_goal_view = bool(self.get_parameter("enable_goal_view").value)

        self._validate_parameters()

        self.depth_enabled = (
            self.oak_with_depth and self.camera_type.lower().strip() == "oak"
        )

        # ---------------------------------------------------------------------
        # ROS publishers/subscribers
        # ---------------------------------------------------------------------

        self.status_topic = f"/perception/{self.camera_name}/status"
        self.detection_topic = f"/perception/{self.camera_name}/detections"
        self.mapped_objects_topic = (
            f"/perception/{self.camera_name}/mapped_objects"
        )
        self.detection_array_topic = f"/perception/{self.camera_name}/detection_array"

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.detection_pub = self.create_publisher(String, self.detection_topic, 10)
        self.mapped_objects_pub = self.create_publisher(
            String,
            self.mapped_objects_topic,
            10,
        )

        if _INTERFACES_AVAILABLE:
            self.detection_array_pub = self.create_publisher(
                DetectionArray,
                self.detection_array_topic,
                10,
            )
        else:
            self.detection_array_pub = None
            self.get_logger().warning(
                "leroi_interfaces not found. DetectionArray will not be published."
            )

        self.odom_sub = self.create_subscription(
            String,
            "/v5/odom",
            self.odom_callback,
            10,
        )

        self.path_sub = self.create_subscription(
            String,
            self.target_path_topic,
            self.target_path_callback,
            10,
        )

        self.action_ack_sub = self.create_subscription(
            String,
            self.action_ack_topic,
            self.action_ack_callback,
            10,
        )

        # ---------------------------------------------------------------------
        # Shared state
        # ---------------------------------------------------------------------

        self.robot_pose_lock = threading.Lock()
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading_deg = 0.0
        self.has_robot_pose = False
        self.last_odom_stamp_ns = 0
        self.robot_trail: List[Tuple[float, float, float, float]] = []

        self.object_map_lock = threading.Lock()
        self.object_history: List[Dict[str, Any]] = []

        self.target_path_lock = threading.Lock()
        self.target_path: List[Dict[str, float]] = []
        self.latest_path_command_id = None
        self.latest_path_action_id = None
        self.latest_path_action_name = None

        self.action_ack_lock = threading.Lock()
        self.latest_action_ack: Optional[Dict[str, Any]] = None

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_depth = None
        self.latest_frame_id = 0

        self.detection_lock = threading.Lock()
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_yolo_frame_id = 0
        self.last_processed_yolo_frame_id = 0

        self.frame_count = 0
        self.dropped_frame_count = 0
        self.yolo_count = 0
        self.yolo_error_count = 0

        self.fps_last_time = time.time()
        self.fps_last_frame_count = 0
        self.fps_last_yolo_count = 0
        self.camera_fps_actual = 0.0
        self.yolo_fps_actual = 0.0

        self.running = True

        # ---------------------------------------------------------------------
        # Camera / detector / streamer
        # ---------------------------------------------------------------------

        self.field_map_background = self.load_field_map_background()

        self.camera = create_camera_source(
            camera_type=self.camera_type,
            camera_index=self.camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            oak_mxid=self.oak_mxid,
            oak_with_depth=self.oak_with_depth,
        )

        self.detector: Optional[YoloDetector] = None

        if self.enable_yolo:
            self.detector = YoloDetector(
                model_path=self.yolo_model_path,
                confidence_threshold=self.yolo_confidence,
                image_size=self.yolo_image_size,
                device=self.yolo_device,
                use_half=self.yolo_use_half,
                class_names=self.yolo_class_names,
            )

        self.preview_streamer: Optional[MjpegPreviewStreamer] = None

        if self.enable_preview_stream:
            self.preview_streamer = MjpegPreviewStreamer(
                host="0.0.0.0",
                port=self.preview_port,
                jpeg_quality=self.preview_jpeg_quality,
                stream_fps=self.preview_stream_fps,
                title=f"Leroi — {self.camera_name}",
                show_map=self.enable_map_stream,
                show_goal=self.enable_goal_view,
            )
            self.preview_streamer.start()
            self.get_logger().info(
                f"Web preview at http://0.0.0.0:{self.preview_port}"
            )

        # ---------------------------------------------------------------------
        # Threads/timers
        # ---------------------------------------------------------------------

        self.camera_thread = threading.Thread(
            target=self.camera_loop,
            name=f"{self.camera_name}_camera_thread",
            daemon=True,
        )
        self.camera_thread.start()

        self.yolo_thread = None

        if self.enable_yolo:
            self.yolo_thread = threading.Thread(
                target=self.yolo_loop,
                name=f"{self.camera_name}_yolo_thread",
                daemon=True,
            )
            self.yolo_thread.start()

        self.preview_timer = self.create_timer(
            1.0 / self.preview_rate_hz,
            self.preview_timer_callback,
        )

        self.status_timer = self.create_timer(
            1.0 / self.status_rate_hz,
            self.status_timer_callback,
        )

        self.get_logger().info("Perception node started.")
        self.get_logger().info(f"Camera: {self.camera_name} ({self.camera_type})")
        self.get_logger().info(f"Depth enabled: {self.depth_enabled}")
        self.get_logger().info(f"YOLO enabled: {self.enable_yolo}")
        self.get_logger().info(f"YOLO target FPS: {self.yolo_rate_hz}")
        self.get_logger().info(f"YOLO model: {self.yolo_model_path}")
        self.get_logger().info(f"Map enabled: {self.enable_map_stream}")
        self.get_logger().info(f"Field map image: {self.field_map_image_path}")
        self.get_logger().info(f"Field map rotation: {self.field_map_rotation}")
        self.get_logger().info(f"Goal view enabled: {self.enable_goal_view}")
        self.get_logger().info(f"Detection topic: {self.detection_topic}")
        self.get_logger().info(
            f"Mapped objects topic: {self.mapped_objects_topic}"
        )

    # -------------------------------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------------------------------

    def _validate_parameters(self):
        if self.camera_fps <= 0:
            raise ValueError("camera_fps must be greater than 0")

        if self.preview_rate_hz <= 0.0:
            raise ValueError("preview_rate_hz must be greater than 0")

        if self.preview_stream_fps <= 0.0:
            raise ValueError("preview_stream_fps must be greater than 0")

        if self.status_rate_hz <= 0.0:
            raise ValueError("status_rate_hz must be greater than 0")

        if self.yolo_rate_hz <= 0.0:
            raise ValueError("yolo_rate_hz must be greater than 0")

        if self.field_max_x_cm <= self.field_min_x_cm:
            raise ValueError("field_max_x_cm must be greater than field_min_x_cm")

        if self.field_max_y_cm <= self.field_min_y_cm:
            raise ValueError("field_max_y_cm must be greater than field_min_y_cm")

        if self.map_width_px <= 0 or self.map_height_px <= 0:
            raise ValueError("map_width_px/map_height_px must be positive")

        if self.field_map_rotation not in ("none", "cw", "ccw", "180"):
            raise ValueError("field_map_rotation must be none, cw, ccw, or 180")

        self.field_map_alpha = max(0.0, min(1.0, self.field_map_alpha))

    def load_field_map_background(self):
        if not self.field_map_image_path:
            return None

        if not os.path.exists(self.field_map_image_path):
            self.get_logger().warning(
                f"Field map image not found: {self.field_map_image_path}. "
                "Using grid-only map."
            )
            return None

        img = cv2.imread(self.field_map_image_path)

        if img is None:
            self.get_logger().warning(
                f"Failed to load field map image: {self.field_map_image_path}. "
                "Using grid-only map."
            )
            return None

        if self.field_map_rotation == "cw":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.field_map_rotation == "ccw":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.field_map_rotation == "180":
            img = cv2.rotate(img, cv2.ROTATE_180)

        self.get_logger().info(
            f"Loaded field map image: {self.field_map_image_path}, "
            f"rotation={self.field_map_rotation}"
        )

        return img

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------

    def odom_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            valid = bool(data.get("valid", True))
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            heading = float(data.get("heading", 0.0))

            if not valid:
                return

            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(heading)
            ):
                return

            with self.robot_pose_lock:
                self.robot_x = x
                self.robot_y = y
                self.robot_heading_deg = heading
                self.has_robot_pose = True
                self.last_odom_stamp_ns = int(data.get("stamp_ns", 0))

                self.robot_trail.append((x, y, heading, time.time()))

                if len(self.robot_trail) > self.robot_trail_max_points:
                    self.robot_trail = self.robot_trail[-self.robot_trail_max_points :]

        except (json.JSONDecodeError, ValueError, TypeError):
            return

    def target_path_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        points = data.get("points", [])
        if not isinstance(points, list):
            return

        normalized = []

        for point in points:
            try:
                if isinstance(point, dict):
                    x = float(point["x"])
                    y = float(point["y"])
                    heading = float(point.get("heading", 0.0))
                elif isinstance(point, list) or isinstance(point, tuple):
                    x = float(point[0])
                    y = float(point[1])
                    heading = float(point[2]) if len(point) >= 3 else 0.0
                else:
                    continue

                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and math.isfinite(heading)
                ):
                    continue

                normalized.append(
                    {
                        "x": x,
                        "y": y,
                        "heading": heading,
                    }
                )
            except (KeyError, ValueError, TypeError, IndexError):
                continue

        with self.target_path_lock:
            self.target_path = normalized
            self.latest_path_command_id = data.get("id")
            self.latest_path_action_id = data.get("action_id")
            self.latest_path_action_name = data.get("action_name")

    def action_ack_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        with self.action_ack_lock:
            self.latest_action_ack = data

    # -------------------------------------------------------------------------
    # Camera / YOLO loops
    # -------------------------------------------------------------------------

    def camera_loop(self):
        target_period = 1.0 / float(self.camera_fps)

        while self.running:
            start_time = time.time()

            frame = self.camera.read()
            depth = self.camera.read_depth() if self.depth_enabled else None

            with self.frame_lock:
                if frame is None:
                    self.dropped_frame_count += 1
                else:
                    self.latest_frame = frame
                    self.frame_count += 1
                    self.latest_frame_id = self.frame_count

                if depth is not None:
                    self.latest_depth = depth

            elapsed = time.time() - start_time
            sleep_time = target_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def yolo_loop(self):
        target_period = 1.0 / float(self.yolo_rate_hz)

        while self.running:
            start_time = time.time()

            if self.detector is None:
                time.sleep(target_period)
                continue

            with self.frame_lock:
                if self.latest_frame is None:
                    frame = None
                    frame_id = 0
                    depth = None
                else:
                    frame = self.latest_frame.copy()
                    frame_id = self.latest_frame_id
                    depth = (
                        self.latest_depth.copy()
                        if self.latest_depth is not None
                        else None
                    )

            if frame is None:
                time.sleep(0.01)
                continue

            if frame_id == self.last_processed_yolo_frame_id:
                time.sleep(0.005)
                continue

            self.last_processed_yolo_frame_id = frame_id

            try:
                intrinsics = getattr(self.camera, "intrinsics", None)

                if self.depth_enabled and depth is not None and intrinsics is not None:
                    fx, fy, cx, cy = intrinsics
                    detections = self.detector.detect_3d(frame, depth, fx, fy, cx, cy)
                else:
                    detections = self.detector.detect(frame)

            except Exception as exc:
                with self.detection_lock:
                    self.yolo_error_count += 1

                self.get_logger().error(f"YOLO inference failed: {exc}")
                time.sleep(0.05)
                continue

            stamp_now = self.get_clock().now()

            with self.robot_pose_lock:
                robot_x = self.robot_x
                robot_y = self.robot_y
                robot_heading_deg = self.robot_heading_deg
                has_robot_pose = self.has_robot_pose

            enriched_detections = []

            packet = {
                "t": stamp_now.nanoseconds,
                "robot_pose_valid": has_robot_pose,
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_heading_deg": robot_heading_deg,
                "detections": [],
            }

            for det in detections:
                d = dict(det)

                if has_robot_pose:
                    self.project_detection_to_field(
                        detection=d,
                        robot_x=robot_x,
                        robot_y=robot_y,
                        robot_heading_deg=robot_heading_deg,
                    )

                x_cm = d.get("x_cm")
                z_cm = d.get("z_cm")
                x_field_cm = d.get("x_field_cm")
                y_field_cm = d.get("y_field_cm")

                packet_detection = {
                    "class": d.get("class_name"),
                    "class_name": d.get("class_name"),
                    "class_id": d.get("class_id"),
                    "x_cm": round(float(x_cm), 1) if x_cm is not None else None,
                    "z_cm": round(float(z_cm), 1) if z_cm is not None else None,
                    "x_field_cm": round(float(x_field_cm), 1)
                    if x_field_cm is not None
                    else None,
                    "y_field_cm": round(float(y_field_cm), 1)
                    if y_field_cm is not None
                    else None,
                    "conf": round(float(d.get("confidence", 0.0)), 2),
                    "confidence": float(d.get("confidence", 0.0)),
                    "bbox_x": d.get("bbox_x"),
                    "bbox_y": d.get("bbox_y"),
                    "bbox_width": d.get("bbox_width"),
                    "bbox_height": d.get("bbox_height"),
                    "bbox_center_x": d.get("bbox_center_x"),
                    "bbox_center_y": d.get("bbox_center_y"),
                }

                packet["detections"].append(packet_detection)
                enriched_detections.append(d)

            with self.detection_lock:
                self.latest_detections = enriched_detections
                self.latest_yolo_frame_id = frame_id
                self.yolo_count += 1

            self.update_object_history(enriched_detections)
            self.publish_mapped_objects(stamp_now.nanoseconds)

            msg = String()
            msg.data = json.dumps(packet)
            self.detection_pub.publish(msg)

            if self.detection_array_pub is not None and _INTERFACES_AVAILABLE:
                self.publish_detection_array(
                    stamp_now=stamp_now,
                    robot_x=robot_x,
                    robot_y=robot_y,
                    robot_heading_deg=robot_heading_deg,
                    detections=enriched_detections,
                )

            elapsed = time.time() - start_time
            sleep_time = target_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # Projection/map helpers
    # -------------------------------------------------------------------------

    def project_detection_to_field(
        self,
        detection: Dict[str, Any],
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
    ):
        """
        Convert camera-relative object coordinates to field coordinates.

        yolo_detector.detect_3d gives:
          x_cm = object right of camera
          z_cm = object forward from camera

        We include camera offset relative to robot center, then rotate by robot heading.

        Robot heading convention:
          0 deg = facing +Y on field.
        """
        x_cm = detection.get("x_cm")
        z_cm = detection.get("z_cm")

        if x_cm is None or z_cm is None:
            return

        try:
            x_cam = float(x_cm)
            z_cam = float(z_cm)

            if not (math.isfinite(x_cam) and math.isfinite(z_cam)):
                return

        except (ValueError, TypeError):
            return

        local_right_cm = self.camera_offset_x_cm + x_cam
        local_forward_cm = self.camera_offset_y_cm + z_cam

        heading_rad = math.radians(robot_heading_deg)

        # Heading 0: forward=(0,+1), right=(+1,0)
        forward_x = math.sin(heading_rad)
        forward_y = math.cos(heading_rad)
        right_x = math.cos(heading_rad)
        right_y = -math.sin(heading_rad)

        x_field = robot_x + local_forward_cm * forward_x + local_right_cm * right_x
        y_field = robot_y + local_forward_cm * forward_y + local_right_cm * right_y

        detection["x_field_cm"] = x_field
        detection["y_field_cm"] = y_field

    def world_to_map_px(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        u = int(
            (x_cm - self.field_min_x_cm)
            / (self.field_max_x_cm - self.field_min_x_cm)
            * self.map_width_px
        )

        v = int(
            self.map_height_px
            - (
                (y_cm - self.field_min_y_cm)
                / (self.field_max_y_cm - self.field_min_y_cm)
                * self.map_height_px
            )
        )

        return u, v

    def in_field_bounds(self, x_cm: float, y_cm: float) -> bool:
        return (
            self.field_min_x_cm <= x_cm <= self.field_max_x_cm
            and self.field_min_y_cm <= y_cm <= self.field_max_y_cm
        )

    def publish_mapped_objects(self, stamp_ns: int):
        """Publish the same field-mapped objects shown on the preview map."""
        with self.object_map_lock:
            objects = list(self.object_history)

        mapped_objects = []

        for obj in objects:
            class_name = str(obj.get("class_name", "?"))
            class_key = class_name.lower()

            try:
                x_field = float(obj["x_field_cm"])
                y_field = float(obj["y_field_cm"])
            except (KeyError, TypeError, ValueError):
                continue

            if not (math.isfinite(x_field) and math.isfinite(y_field)):
                continue

            color = "unknown"
            if "blue" in class_key:
                color = "blue"
            elif "red" in class_key:
                color = "red"

            mapped_objects.append(
                {
                    "id": f"{class_key}-{round(x_field)}-{round(y_field)}",
                    "class_name": class_name,
                    "x_field_cm": round(x_field, 1),
                    "y_field_cm": round(y_field, 1),
                    "color": color,
                    "confidence": obj.get("confidence"),
                }
            )

        packet = {
            "t": stamp_ns,
            "object_count": len(mapped_objects),
            "objects": mapped_objects,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.mapped_objects_pub.publish(msg)

    def update_object_history(self, detections: List[Dict[str, Any]]):
        now = time.time()

        with self.object_map_lock:
            for d in detections:
                x_field = d.get("x_field_cm")
                y_field = d.get("y_field_cm")

                if x_field is None or y_field is None:
                    continue

                try:
                    x_field = float(x_field)
                    y_field = float(y_field)
                except (ValueError, TypeError):
                    continue

                if not (math.isfinite(x_field) and math.isfinite(y_field)):
                    continue

                if not self.in_field_bounds(x_field, y_field):
                    continue

                x_cm = d.get("x_cm")
                z_cm = d.get("z_cm")

                self.object_history.append(
                    {
                        "time": now,
                        "class_name": str(d.get("class_name", "?")),
                        "confidence": float(d.get("confidence", 0.0)),
                        "x_cm": float(x_cm) if x_cm is not None else None,
                        "z_cm": float(z_cm) if z_cm is not None else None,
                        "x_field_cm": x_field,
                        "y_field_cm": y_field,
                    }
                )

            cutoff = now - self.object_history_sec
            self.object_history = [
                item for item in self.object_history if item["time"] >= cutoff
            ]

    # -------------------------------------------------------------------------
    # Preview/status timers
    # -------------------------------------------------------------------------

    def preview_timer_callback(self):
        with self.frame_lock:
            if self.latest_frame is None:
                frame = None
                frame_id = self.latest_frame_id
            else:
                frame = self.latest_frame.copy()
                frame_id = self.latest_frame_id

        with self.detection_lock:
            detections = list(self.latest_detections)
            yolo_frame_id = self.latest_yolo_frame_id

        if frame is not None:
            if self.draw_detections:
                self.draw_detection_overlay(
                    frame=frame,
                    detections=detections,
                    frame_id=frame_id,
                    yolo_frame_id=yolo_frame_id,
                )

            if self.preview_streamer is not None:
                self.preview_streamer.update_frame(frame, frame_id=frame_id)

        if self.preview_streamer is not None and self.enable_map_stream:
            map_img = self.draw_field_map()
            self.preview_streamer.update_map_frame(map_img, frame_id=frame_id)

    def status_timer_callback(self):
        self.update_fps_estimates()

        with self.frame_lock:
            frame_count = self.frame_count
            latest_frame_id = self.latest_frame_id
            dropped_frame_count = self.dropped_frame_count

        with self.detection_lock:
            detection_count = len(self.latest_detections)
            yolo_frame_id = self.latest_yolo_frame_id
            yolo_count = self.yolo_count
            yolo_error_count = self.yolo_error_count

        with self.robot_pose_lock:
            robot_pose = {
                "valid": self.has_robot_pose,
                "x": self.robot_x,
                "y": self.robot_y,
                "heading": self.robot_heading_deg,
            }

        frame_lag = max(0, latest_frame_id - yolo_frame_id)

        msg = String()
        msg.data = json.dumps(
            {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "camera_name": self.camera_name,
                "camera_type": self.camera_type,
                "status": "running",
                "camera_fps_actual": self.camera_fps_actual,
                "yolo_fps_actual": self.yolo_fps_actual,
                "frame_count": frame_count,
                "latest_frame_id": latest_frame_id,
                "dropped_frame_count": dropped_frame_count,
                "yolo_enabled": self.enable_yolo,
                "yolo_device": self.yolo_device,
                "yolo_count": yolo_count,
                "yolo_error_count": yolo_error_count,
                "latest_yolo_frame_id": yolo_frame_id,
                "yolo_frame_lag": frame_lag,
                "latest_detection_count": detection_count,
                "depth_enabled": self.depth_enabled,
                "preview_enabled": self.enable_preview_stream,
                "map_enabled": self.enable_map_stream,
                "field_map_image_path": self.field_map_image_path,
                "field_map_rotation": self.field_map_rotation,
                "robot_pose": robot_pose,
            }
        )
        self.status_pub.publish(msg)

        self.get_logger().info(
            f"{self.camera_name}: "
            f"cam_fps={self.camera_fps_actual:.1f}, "
            f"yolo_fps={self.yolo_fps_actual:.1f}, "
            f"frames={frame_count}, "
            f"dropped={dropped_frame_count}, "
            f"yolo_runs={yolo_count}, "
            f"yolo_errors={yolo_error_count}, "
            f"frame_lag={frame_lag}, "
            f"detections={detection_count}, "
            f"robot_pose_valid={robot_pose['valid']}"
        )

    def update_fps_estimates(self):
        now = time.time()
        dt = now - self.fps_last_time

        if dt < 0.5:
            return

        with self.frame_lock:
            frame_count = self.frame_count

        with self.detection_lock:
            yolo_count = self.yolo_count

        frame_delta = frame_count - self.fps_last_frame_count
        yolo_delta = yolo_count - self.fps_last_yolo_count

        self.camera_fps_actual = frame_delta / dt
        self.yolo_fps_actual = yolo_delta / dt

        self.fps_last_time = now
        self.fps_last_frame_count = frame_count
        self.fps_last_yolo_count = yolo_count

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def draw_field_map(self):
        w = self.map_width_px
        h = self.map_height_px

        if self.field_map_background is not None:
            bg = cv2.resize(self.field_map_background, (w, h))
            dark = np.zeros_like(bg)
            img = cv2.addWeighted(
                bg,
                self.field_map_alpha,
                dark,
                1.0 - self.field_map_alpha,
                0,
            )
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = (25, 25, 25)

        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (230, 230, 230), 2)

        if self.draw_field_grid:
            for i in range(1, 6):
                x = int(w * i / 6)
                y = int(h * i / 6)
                cv2.line(img, (x, 0), (x, h), (80, 80, 80), 1)
                cv2.line(img, (0, y), (w, y), (80, 80, 80), 1)

        with self.robot_pose_lock:
            robot_x = self.robot_x
            robot_y = self.robot_y
            robot_heading = self.robot_heading_deg
            has_robot_pose = self.has_robot_pose
            trail = list(self.robot_trail)

        with self.target_path_lock:
            target_path = list(self.target_path)
            path_id = self.latest_path_command_id
            path_action_id = self.latest_path_action_id
            path_action_name = self.latest_path_action_name

        with self.object_map_lock:
            objects = list(self.object_history)

        with self.action_ack_lock:
            latest_ack = dict(self.latest_action_ack) if self.latest_action_ack else None

        # Draw target path
        if target_path:
            path_points_px = []

            for p in target_path:
                px = self.world_to_map_px(float(p["x"]), float(p["y"]))
                path_points_px.append(px)

            for i in range(len(path_points_px) - 1):
                cv2.line(
                    img,
                    path_points_px[i],
                    path_points_px[i + 1],
                    (0, 180, 255),
                    2,
                )

            for i, px in enumerate(path_points_px):
                cv2.circle(img, px, 6, (0, 180, 255), -1)
                cv2.putText(
                    img,
                    str(i + 1),
                    (px[0] + 6, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 180, 255),
                    1,
                )

        # Draw robot trail
        if len(trail) >= 2:
            trail_points = [self.world_to_map_px(p[0], p[1]) for p in trail]

            for i in range(len(trail_points) - 1):
                cv2.line(
                    img,
                    trail_points[i],
                    trail_points[i + 1],
                    (90, 120, 255),
                    2,
                )

        # Draw projected detections
        now = time.time()

        for obj in objects:
            x = float(obj["x_field_cm"])
            y = float(obj["y_field_cm"])
            age = now - float(obj["time"])
            px = self.world_to_map_px(x, y)

            if px[0] < 0 or px[0] >= w or px[1] < 0 or px[1] >= h:
                continue

            intensity = max(
                60,
                int(255 * (1.0 - age / max(self.object_history_sec, 0.1))),
            )

            class_name = str(obj.get("class_name", "?"))
            color = YoloDetector.class_color(class_name)

            # Fade color with age
            color = tuple(int(c * intensity / 255) for c in color)

            cv2.circle(img, px, 6, color, -1)

            label = class_name[:10]

            if obj.get("z_cm") is not None:
                label += f" {float(obj['z_cm']):.0f}cm"

            cv2.putText(
                img,
                label,
                (px[0] + 7, px[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
            )

        # Draw robot pose
        if has_robot_pose:
            robot_px = self.world_to_map_px(robot_x, robot_y)

            heading_rad = math.radians(robot_heading)
            length = 24
            width_px = 13

            nose = (
                int(robot_px[0] + length * math.sin(heading_rad)),
                int(robot_px[1] - length * math.cos(heading_rad)),
            )

            left = (
                int(robot_px[0] - width_px * math.cos(heading_rad)),
                int(robot_px[1] - width_px * math.sin(heading_rad)),
            )

            right = (
                int(robot_px[0] + width_px * math.cos(heading_rad)),
                int(robot_px[1] + width_px * math.sin(heading_rad)),
            )

            triangle = np.array([nose, left, right], dtype=np.int32)
            cv2.fillPoly(img, [triangle], (255, 255, 255))
            cv2.circle(img, robot_px, 4, (0, 0, 255), -1)

        # HUD background
        cv2.rectangle(img, (0, 0), (w, 108), (10, 10, 10), -1)

        cv2.putText(
            img,
            "Leroi Field Map",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
        )

        if has_robot_pose:
            pose_text = (
                f"Robot: x={robot_x:.1f} y={robot_y:.1f} h={robot_heading:.1f}"
            )
        else:
            pose_text = "Robot: no valid odom"

        cv2.putText(
            img,
            pose_text,
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

        path_text = (
            f"Cmd: id={path_id} action={path_action_id} "
            f"{path_action_name or ''} points={len(target_path)}"
        )

        cv2.putText(
            img,
            path_text[:90],
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 180, 255),
            1,
        )

        obj_text = f"Objects projected: {len(objects)}"
        cv2.putText(
            img,
            obj_text,
            (12, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 0),
            1,
        )

        if latest_ack:
            ack_text = (
                f"ACK id={latest_ack.get('command_id')} "
                f"{latest_ack.get('status')} "
                f"action={latest_ack.get('action_id')} "
                f"{latest_ack.get('detail', '')}"
            )
            cv2.putText(
                img,
                ack_text[:95],
                (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        return img

    def draw_detection_overlay(self, frame, detections, frame_id, yolo_frame_id):
        for det in detections:
            x = int(det.get("bbox_x", 0))
            y = int(det.get("bbox_y", 0))
            w = int(det.get("bbox_width", 0))
            h = int(det.get("bbox_height", 0))

            class_name = str(det.get("class_name", "object"))
            confidence = float(det.get("confidence", 0.0))

            color = YoloDetector.class_color(class_name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            x_cm = det.get("x_cm")
            z_cm = det.get("z_cm")
            x_field_cm = det.get("x_field_cm")
            y_field_cm = det.get("y_field_cm")

            if x_cm is not None and z_cm is not None:
                label = (
                    f"{class_name} {confidence:.2f} "
                    f"X:{float(x_cm):.1f} Z:{float(z_cm):.1f}cm"
                )
            else:
                label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if x_field_cm is not None and y_field_cm is not None:
                field_label = (
                    f"field ({float(x_field_cm):.1f}, {float(y_field_cm):.1f})"
                )
                cv2.putText(
                    frame,
                    field_label,
                    (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

            depth_region = det.get("depth_region")
            if depth_region is not None:
                rx1, ry1, rx2, ry2 = depth_region
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)

        frame_lag = max(0, frame_id - yolo_frame_id)

        cv2.putText(
            frame,
            f"{self.camera_name} | cam {self.camera_fps_actual:.1f} FPS | frame {frame_id}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"YOLO {self.yolo_fps_actual:.1f} FPS | frame {yolo_frame_id} | lag {frame_lag} | det {len(detections)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        with self.robot_pose_lock:
            has_robot_pose = self.has_robot_pose
            robot_x = self.robot_x
            robot_y = self.robot_y
            robot_heading = self.robot_heading_deg

        if has_robot_pose:
            odom_text = (
                f"ODOM x={robot_x:.1f}cm y={robot_y:.1f}cm h={robot_heading:.1f}deg"
            )
            odom_color = (0, 255, 0)
        else:
            odom_text = "ODOM: no valid V5 pose"
            odom_color = (0, 0, 255)

        cv2.putText(
            frame,
            odom_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            odom_color,
            2,
        )

    # -------------------------------------------------------------------------
    # Optional typed interface publisher
    # -------------------------------------------------------------------------

    def publish_detection_array(
        self,
        stamp_now,
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
        detections: List[Dict[str, Any]],
    ):
        det_array = DetectionArray()
        det_array.header.stamp = stamp_now.to_msg()
        det_array.header.frame_id = self.camera_name
        det_array.camera_name = self.camera_name
        det_array.robot_x = float(robot_x)
        det_array.robot_y = float(robot_y)
        det_array.robot_heading_deg = float(robot_heading_deg)

        for d in detections:
            obj = DetectedObject()
            obj.stamp = stamp_now.to_msg()
            obj.class_name = str(d.get("class_name", ""))
            obj.class_id = int(d.get("class_id", 0))
            obj.confidence = float(d.get("confidence", 0.0))
            obj.bbox_x = float(d.get("bbox_x", 0.0))
            obj.bbox_y = float(d.get("bbox_y", 0.0))
            obj.bbox_width = float(d.get("bbox_width", 0.0))
            obj.bbox_height = float(d.get("bbox_height", 0.0))

            x_cm = d.get("x_cm")
            z_cm = d.get("z_cm")
            x_field_cm = d.get("x_field_cm")
            y_field_cm = d.get("y_field_cm")

            obj.x_cm = float(x_cm) if x_cm is not None else float("nan")
            obj.z_cm = float(z_cm) if z_cm is not None else float("nan")
            obj.x_field_cm = (
                float(x_field_cm) if x_field_cm is not None else float("nan")
            )
            obj.y_field_cm = (
                float(y_field_cm) if y_field_cm is not None else float("nan")
            )

            det_array.objects.append(obj)

        self.detection_array_pub.publish(det_array)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def destroy_node(self):
        self.running = False

        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)

        if self.yolo_thread is not None and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=1.0)

        if self.preview_streamer is not None:
            self.preview_streamer.stop()

        if self.camera is not None:
            try:
                self.camera.close()
            except AttributeError:
                pass
            except Exception:
                pass
            self.camera = None

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Perception node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()