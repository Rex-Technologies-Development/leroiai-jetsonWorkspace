import json
import math
import threading
import time
from typing import Optional

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

try:
    from leroi_perception.goal_projector import (
        GoalProjector,
        make_vex_high_goal_corners,
    )

    _PROJECTOR_AVAILABLE = True
except ImportError:
    _PROJECTOR_AVAILABLE = False


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

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
        self.declare_parameter("field_min_x_cm", -183.0)
        self.declare_parameter("field_max_x_cm", 183.0)
        self.declare_parameter("field_min_y_cm", -183.0)
        self.declare_parameter("field_max_y_cm", 183.0)
        self.declare_parameter("robot_trail_max_points", 500)
        self.declare_parameter("object_history_sec", 3.0)
        self.declare_parameter("target_path_topic", "/robot/path_command")
        self.declare_parameter("action_ack_topic", "/robot/action_ack")

        # Goal view parameters
        self.declare_parameter("enable_goal_view", False)
        self.declare_parameter("goal_rect_x1", 60)
        self.declare_parameter("goal_rect_y1", 120)
        self.declare_parameter("goal_rect_x2", 580)
        self.declare_parameter("goal_rect_y2", 400)

        # Goal projector parameters
        self.declare_parameter("goal_center_x_cm", 0.0)
        self.declare_parameter("goal_center_y_cm", 183.0)
        self.declare_parameter("goal_z_bottom_cm", 50.0)
        self.declare_parameter("goal_width_cm", 61.0)
        self.declare_parameter("goal_height_cm", 30.0)

        # Camera mount parameters
        self.declare_parameter("camera_offset_x_cm", 0.0)
        self.declare_parameter("camera_offset_y_cm", 15.0)
        self.declare_parameter("camera_offset_z_cm", 25.0)
        self.declare_parameter("camera_tilt_deg", 10.0)

        # Hardcoded fallback pose for testing goal view
        self.declare_parameter("test_robot_x", 0.0)
        self.declare_parameter("test_robot_y", 0.0)
        self.declare_parameter("test_robot_heading_deg", 0.0)

        # Read camera parameters
        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_type = str(self.get_parameter("camera_type").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.oak_mxid = str(self.get_parameter("oak_mxid").value)
        self.oak_with_depth = bool(self.get_parameter("oak_with_depth").value)

        # Read preview/status parameters
        self.preview_rate_hz = float(self.get_parameter("preview_rate_hz").value)
        self.preview_stream_fps = float(self.get_parameter("preview_stream_fps").value)
        self.status_rate_hz = float(self.get_parameter("status_rate_hz").value)

        # Read YOLO parameters
        self.enable_yolo = bool(self.get_parameter("enable_yolo").value)
        self.yolo_rate_hz = float(self.get_parameter("yolo_rate_hz").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_confidence = float(self.get_parameter("yolo_confidence").value)
        self.yolo_image_size = int(self.get_parameter("yolo_image_size").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_use_half = bool(self.get_parameter("yolo_use_half").value)

        names_raw = str(self.get_parameter("yolo_class_names").value).strip()
        if names_raw:
            try:
                parsed = json.loads(names_raw)
                self.yolo_class_names = {int(k): str(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, ValueError):
                self.get_logger().warning(
                    "yolo_class_names is not valid JSON. Ignoring."
                )
                self.yolo_class_names = None
        else:
            self.yolo_class_names = None

        # Read web preview parameters
        self.enable_preview_stream = bool(
            self.get_parameter("enable_preview_stream").value
        )
        self.preview_port = int(self.get_parameter("preview_port").value)
        self.preview_jpeg_quality = int(
            self.get_parameter("preview_jpeg_quality").value
        )
        self.draw_detections = bool(self.get_parameter("draw_detections").value)

        # Read map parameters
        self.enable_map_stream = bool(self.get_parameter("enable_map_stream").value)
        self.map_width_px = int(self.get_parameter("map_width_px").value)
        self.map_height_px = int(self.get_parameter("map_height_px").value)
        self.field_min_x_cm = float(self.get_parameter("field_min_x_cm").value)
        self.field_max_x_cm = float(self.get_parameter("field_max_x_cm").value)
        self.field_min_y_cm = float(self.get_parameter("field_min_y_cm").value)
        self.field_max_y_cm = float(self.get_parameter("field_max_y_cm").value)
        self.robot_trail_max_points = int(
            self.get_parameter("robot_trail_max_points").value
        )
        self.object_history_sec = float(self.get_parameter("object_history_sec").value)
        self.target_path_topic = str(self.get_parameter("target_path_topic").value)
        self.action_ack_topic = str(self.get_parameter("action_ack_topic").value)

        # Read goal view parameters
        self.enable_goal_view = bool(self.get_parameter("enable_goal_view").value)
        self.goal_rect = (
            int(self.get_parameter("goal_rect_x1").value),
            int(self.get_parameter("goal_rect_y1").value),
            int(self.get_parameter("goal_rect_x2").value),
            int(self.get_parameter("goal_rect_y2").value),
        )

        # Read goal projector parameters
        self._goal_center_x = float(self.get_parameter("goal_center_x_cm").value)
        self._goal_center_y = float(self.get_parameter("goal_center_y_cm").value)
        self._goal_z_bottom = float(self.get_parameter("goal_z_bottom_cm").value)
        self._goal_width = float(self.get_parameter("goal_width_cm").value)
        self._goal_height = float(self.get_parameter("goal_height_cm").value)

        self._cam_offset = (
            float(self.get_parameter("camera_offset_x_cm").value),
            float(self.get_parameter("camera_offset_y_cm").value),
            float(self.get_parameter("camera_offset_z_cm").value),
        )
        self._cam_tilt = float(self.get_parameter("camera_tilt_deg").value)

        self._test_robot_x = float(self.get_parameter("test_robot_x").value)
        self._test_robot_y = float(self.get_parameter("test_robot_y").value)
        self._test_heading = float(self.get_parameter("test_robot_heading_deg").value)

        self._validate_parameters()

        # Depth is only meaningful on OAK cameras.
        self.depth_enabled = (
            self.oak_with_depth and self.camera_type.lower().strip() == "oak"
        )

        # ROS topics
        self.status_topic = f"/perception/{self.camera_name}/status"
        self.detection_topic = f"/perception/{self.camera_name}/detections"
        self.detection_array_topic = f"/perception/{self.camera_name}/detection_array"

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.detection_pub = self.create_publisher(String, self.detection_topic, 10)

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

        # Robot pose/map state
        self.robot_pose_lock = threading.Lock()
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading_deg = 0.0
        self.has_robot_pose = False
        self.last_odom_stamp_ns = 0
        self.robot_trail = []

        self.object_map_lock = threading.Lock()
        self.object_history = []

        self.target_path_lock = threading.Lock()
        self.target_path = []
        self.latest_path_command_id = None

        self.action_ack_lock = threading.Lock()
        self.latest_action_ack = None

        # Subscribers
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

        # Camera source
        self.camera = create_camera_source(
            camera_type=self.camera_type,
            camera_index=self.camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            oak_mxid=self.oak_mxid,
            oak_with_depth=self.oak_with_depth,
        )

        # Goal projector
        self.goal_projector = None
        if _PROJECTOR_AVAILABLE and self.enable_goal_view:
            intrinsics = getattr(self.camera, "intrinsics", None)
            if intrinsics is not None:
                fx, fy, cx_i, cy_i = intrinsics
                corners = make_vex_high_goal_corners(
                    self._goal_center_x,
                    self._goal_center_y,
                    self._goal_z_bottom,
                    self._goal_width,
                    self._goal_height,
                )

                self.goal_projector = GoalProjector(
                    fx=fx,
                    fy=fy,
                    cx=cx_i,
                    cy=cy_i,
                    goal_corners_world=corners,
                    camera_offset_cm=self._cam_offset,
                    camera_tilt_deg=self._cam_tilt,
                )

                self.get_logger().info(
                    "GoalProjector initialized with camera intrinsics."
                )
            else:
                self.get_logger().warning(
                    "GoalProjector: camera intrinsics not available. "
                    "Projection disabled."
                )

        # YOLO detector
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

        # Single web streamer on one port.
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

        # Tape divider detection state
        self._tape_x1: Optional[float] = None
        self._tape_x2: Optional[float] = None
        self._tape_alpha = 0.15

        # Locks
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()

        # Shared camera frame state
        self.latest_frame = None
        self.latest_depth = None
        self.latest_frame_id = 0

        # Shared detection state
        self.latest_detections = []
        self.latest_yolo_frame_id = 0
        self.last_processed_yolo_frame_id = 0

        # Counters
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.yolo_count = 0
        self.yolo_error_count = 0

        # FPS tracking
        self.fps_last_time = time.time()
        self.fps_last_frame_count = 0
        self.fps_last_yolo_count = 0
        self.camera_fps_actual = 0.0
        self.yolo_fps_actual = 0.0

        # Threads
        self.running = True

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

        # ROS timers
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
        self.get_logger().info(f"Camera target FPS: {self.camera_fps}")
        self.get_logger().info(f"YOLO enabled: {self.enable_yolo}")
        self.get_logger().info(f"YOLO target FPS: {self.yolo_rate_hz}")
        self.get_logger().info(f"YOLO model: {self.yolo_model_path}")
        self.get_logger().info(f"Map enabled: {self.enable_map_stream}")
        self.get_logger().info(f"Goal view enabled: {self.enable_goal_view}")
        self.get_logger().info(f"Status topic: {self.status_topic}")
        self.get_logger().info(f"Detection topic: {self.detection_topic}")

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

    def action_ack_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        with self.action_ack_lock:
            self.latest_action_ack = data

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

            heading_rad = math.radians(robot_heading_deg)
            enriched_detections = []

            packet = {
                "t": stamp_now.nanoseconds,
                "robot_pose_valid": has_robot_pose,
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_heading_deg": robot_heading_deg,
                "detections": [],
            }

            for d in detections:
                d = dict(d)

                x_cm = d.get("x_cm")
                z_cm = d.get("z_cm")

                x_field_cm = None
                y_field_cm = None

                if has_robot_pose and x_cm is not None and z_cm is not None:
                    try:
                        x_cm_f = float(x_cm)
                        z_cm_f = float(z_cm)

                        x_field_cm = (
                            robot_x
                            + z_cm_f * math.cos(heading_rad)
                            - x_cm_f * math.sin(heading_rad)
                        )
                        y_field_cm = (
                            robot_y
                            + z_cm_f * math.sin(heading_rad)
                            + x_cm_f * math.cos(heading_rad)
                        )

                        d["x_field_cm"] = x_field_cm
                        d["y_field_cm"] = y_field_cm

                    except (ValueError, TypeError):
                        pass

                packet_detection = {
                    "class": d.get("class_name"),
                    "class_name": d.get("class_name"),
                    "class_id": d.get("class_id"),
                    "x_cm": round(float(x_cm), 1) if x_cm is not None else None,
                    "z_cm": round(float(z_cm), 1) if z_cm is not None else None,
                    "x_field_cm": round(x_field_cm, 1)
                    if x_field_cm is not None
                    else None,
                    "y_field_cm": round(y_field_cm, 1)
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

            msg = String()
            msg.data = json.dumps(packet)
            self.detection_pub.publish(msg)

            if self.detection_array_pub is not None and _INTERFACES_AVAILABLE:
                self.publish_detection_array(
                    stamp_now,
                    robot_x,
                    robot_y,
                    robot_heading_deg,
                    enriched_detections,
                )

            elapsed = time.time() - start_time
            sleep_time = target_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def publish_detection_array(
        self,
        stamp_now,
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
        detections,
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

    def update_object_history(self, detections):
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

                self.object_history.append(
                    {
                        "time": now,
                        "class_name": str(d.get("class_name", "?")),
                        "confidence": float(d.get("confidence", 0.0)),
                        "x_field_cm": x_field,
                        "y_field_cm": y_field,
                    }
                )

            cutoff = now - self.object_history_sec
            self.object_history = [
                item for item in self.object_history if item["time"] >= cutoff
            ]

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

                if self.enable_goal_view:
                    goal_img = self.draw_goal_view(frame.copy(), detections)
                    self.preview_streamer.update_goal_frame(
                        goal_img,
                        frame_id=frame_id,
                    )

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
                "goal_view_enabled": self.enable_goal_view,
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

    def world_to_map_px(self, x_cm: float, y_cm: float):
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

    def draw_field_map(self):
        w = self.map_width_px
        h = self.map_height_px

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (25, 25, 25)

        # Field border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (180, 180, 180), 2)

        # Grid
        for i in range(1, 6):
            x = int(w * i / 6)
            y = int(h * i / 6)
            cv2.line(img, (x, 0), (x, h), (45, 45, 45), 1)
            cv2.line(img, (0, y), (w, y), (45, 45, 45), 1)

        with self.robot_pose_lock:
            robot_x = self.robot_x
            robot_y = self.robot_y
            robot_heading = self.robot_heading_deg
            has_robot_pose = self.has_robot_pose
            trail = list(self.robot_trail)

        with self.target_path_lock:
            target_path = list(self.target_path)
            path_id = self.latest_path_command_id

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
                cv2.line(img, path_points_px[i], path_points_px[i + 1], (0, 180, 255), 2)

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
                cv2.line(img, trail_points[i], trail_points[i + 1], (90, 120, 255), 2)

        # Draw object detections
        now = time.time()
        for obj in objects:
            x = obj["x_field_cm"]
            y = obj["y_field_cm"]
            age = now - obj["time"]
            px = self.world_to_map_px(x, y)

            intensity = max(60, int(255 * (1.0 - age / max(self.object_history_sec, 0.1))))
            color = (0, intensity, 0)

            cv2.circle(img, px, 5, color, -1)
            cv2.putText(
                img,
                obj["class_name"][:10],
                (px[0] + 6, px[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        # Draw robot pose
        if has_robot_pose:
            robot_px = self.world_to_map_px(robot_x, robot_y)

            heading_rad = math.radians(robot_heading)
            length = 22
            width_px = 12

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

        # HUD
        cv2.rectangle(img, (0, 0), (w, 88), (10, 10, 10), -1)

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
            pose_text = f"Robot: x={robot_x:.1f} y={robot_y:.1f} h={robot_heading:.1f}"
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

        path_text = f"Path ID: {path_id} points={len(target_path)}"
        cv2.putText(
            img,
            path_text,
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            1,
        )

        if latest_ack:
            ack_text = (
                f"ACK id={latest_ack.get('command_id')} "
                f"{latest_ack.get('status')} {latest_ack.get('detail', '')}"
            )
            cv2.putText(
                img,
                ack_text[:80],
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

    def _detect_tape_dividers(self, frame) -> Optional[tuple]:
        gx1, gy1, gx2, gy2 = self.goal_rect
        roi = frame[gy1:gy2, gx1:gx2]

        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))

        col_sums = np.sum(white_mask, axis=0).astype(np.float32)
        kernel = np.ones(9, np.float32) / 9
        col_sums_smooth = np.convolve(col_sums, kernel, mode="same")

        goal_w = gx2 - gx1
        min_peak_sep = goal_w // 5
        margin = goal_w // 10

        search = col_sums_smooth.copy()
        search[:margin] = 0
        search[-margin:] = 0

        peak1_local = int(np.argmax(search))
        peak1_val = search[peak1_local]

        suppress_lo = max(0, peak1_local - min_peak_sep)
        suppress_hi = min(goal_w, peak1_local + min_peak_sep)
        search[suppress_lo:suppress_hi] = 0

        peak2_local = int(np.argmax(search))
        peak2_val = search[peak2_local]

        min_white_rows = (gy2 - gy1) * 0.15
        if peak1_val < min_white_rows or peak2_val < min_white_rows:
            return None

        x1 = gx1 + min(peak1_local, peak2_local)
        x2 = gx1 + max(peak1_local, peak2_local)

        return x1, x2

    def draw_goal_view(self, frame, detections):
        gx1, gy1, gx2, gy2 = self.goal_rect
        goal_w = gx2 - gx1

        detected = self._detect_tape_dividers(frame)
        tape_detected = False

        if detected is not None:
            tx1, tx2 = detected

            if self._tape_x1 is None:
                self._tape_x1 = float(tx1)
                self._tape_x2 = float(tx2)
            else:
                a = self._tape_alpha
                self._tape_x1 = self._tape_x1 * (1 - a) + tx1 * a
                self._tape_x2 = self._tape_x2 * (1 - a) + tx2 * a

            tape_detected = True

        if self._tape_x1 is not None:
            third1_x = int(self._tape_x1)
            third2_x = int(self._tape_x2)
        else:
            third1_x = gx1 + goal_w // 3
            third2_x = gx1 + (2 * goal_w) // 3

        overlay = frame.copy()
        overlay[:, :] = (overlay[:, :] * 0.35).astype("uint8")
        overlay[gy1:gy2, gx1:gx2] = frame[gy1:gy2, gx1:gx2]
        frame = overlay

        if self.goal_projector is not None:
            with self.robot_pose_lock:
                if self.has_robot_pose:
                    rx = self.robot_x
                    ry = self.robot_y
                    rh = self.robot_heading_deg
                else:
                    rx = self._test_robot_x
                    ry = self._test_robot_y
                    rh = self._test_heading

            frame = self.goal_projector.draw_overlay(frame, rx, ry, rh, detections)
            cv2.putText(
                frame,
                "PROJECTION MODE",
                (gx1 + 4, gy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            return frame

        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 255, 255), 2)

        status_text = "TAPE DETECTED" if tape_detected else "GEOMETRIC FALLBACK"
        status_color = (0, 255, 0) if tape_detected else (0, 165, 255)

        cv2.putText(
            frame,
            status_text,
            (gx1 + 4, gy1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
        )

        cv2.line(frame, (third1_x, gy1), (third1_x, gy2), (200, 200, 200), 2)
        cv2.line(frame, (third2_x, gy1), (third2_x, gy2), (200, 200, 200), 2)

        label_y = gy1 + 24
        for label, lx in [
            ("LEFT", gx1 + 8),
            ("MIDDLE", third1_x + 8),
            ("RIGHT", third2_x + 8),
        ]:
            cv2.putText(
                frame,
                label,
                (lx, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

        in_goal = [
            d for d in detections if "in_goal" in str(d.get("class_name", ""))
        ]

        thirds_count = {"LEFT": [], "MIDDLE": [], "RIGHT": []}

        for det in in_goal:
            cx = float(
                det.get(
                    "bbox_center_x",
                    det.get("bbox_x", 0) + det.get("bbox_width", 0) / 2,
                )
            )
            x = int(det.get("bbox_x", 0))
            y = int(det.get("bbox_y", 0))
            w = int(det.get("bbox_width", 0))
            h = int(det.get("bbox_height", 0))
            name = str(det.get("class_name", ""))
            conf = float(det.get("confidence", 0.0))
            color = YoloDetector.class_color(name)

            if cx < third1_x:
                third = "LEFT"
            elif cx < third2_x:
                third = "MIDDLE"
            else:
                third = "RIGHT"

            thirds_count[third].append(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{third} {conf:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        summary_parts = []
        for third in ("LEFT", "MIDDLE", "RIGHT"):
            items = thirds_count[third]

            if items:
                counts = {}
                for n in items:
                    counts[n] = counts.get(n, 0) + 1

                summary_parts.append(
                    f"{third}: " + ", ".join(f"{v}x {k}" for k, v in counts.items())
                )
            else:
                summary_parts.append(f"{third}: --")

        summary = " | ".join(summary_parts)

        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 32),
            (frame.shape[1], frame.shape[0]),
            (30, 30, 30),
            -1,
        )

        cv2.putText(
            frame,
            summary,
            (8, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )

        return frame

    def destroy_node(self):
        self.running = False

        if hasattr(self, "camera_thread") and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)

        if self.yolo_thread is not None and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=1.0)

        if self.preview_streamer is not None:
            self.preview_streamer.stop()
            self.preview_streamer = None

        if self.camera is not None:
            self.camera.close()
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