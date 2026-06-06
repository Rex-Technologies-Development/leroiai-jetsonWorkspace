import json
import math
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from leroi_dashboard.leroi_dashboard import (
    connect_blocks,
    connect_camera,
    connect_robot,
    resolve_host,
    send_blocks,
    send_frame,
    send_location,
)


class DashboardNode(Node):
    def __init__(self):
        super().__init__("dashboard_node")

        self.declare_parameter("enable_dashboard", True)
        self.declare_parameter("bot_id", "")
        self.declare_parameter("vexai_host", "")
        self.declare_parameter("odom_topic", "/v5/odom")
        self.declare_parameter("detection_topic", "/perception/oak_front/detections")
        self.declare_parameter(
            "mapped_objects_topic",
            "/perception/oak_front/mapped_objects",
        )
        self.declare_parameter("action_ack_topic", "/robot/action_ack")
        self.declare_parameter("status_topic", "/dashboard/status")
        self.declare_parameter("sync_rate_hz", 2.0)
        self.declare_parameter("reconnect_period_sec", 5.0)
        self.declare_parameter("connect_log_interval_sec", 60.0)
        self.declare_parameter("field_cm_to_dashboard", 1.0)
        self.declare_parameter("send_all_field_detections", True)
        self.declare_parameter("block_class_substrings", ["block", "red", "blue"])
        self.declare_parameter("object_history_sec", 3.0)
        self.declare_parameter("camera_offset_x_cm", 0.0)
        self.declare_parameter("camera_offset_y_cm", 15.0)
        self.declare_parameter("camera_offset_z_cm", 25.0)
        self.declare_parameter("enable_robot_channel", True)
        self.declare_parameter("enable_blocks_channel", True)
        self.declare_parameter("enable_camera_channel", True)
        self.declare_parameter("camera_id", "cam-1")
        self.declare_parameter("camera_label", "oak_front")
        self.declare_parameter("camera_stream_fps", 10.0)
        self.declare_parameter("camera_preview_url", "http://127.0.0.1:8080/camera.jpg")

        self.enable_dashboard = self._parse_bool(
            self.get_parameter("enable_dashboard").value,
            default=True,
        )
        self.bot_id = str(self.get_parameter("bot_id").value).strip()
        self.vexai_host = str(self.get_parameter("vexai_host").value).strip() or None
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.mapped_objects_topic = str(
            self.get_parameter("mapped_objects_topic").value
        )
        self.action_ack_topic = str(self.get_parameter("action_ack_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.sync_rate_hz = float(self.get_parameter("sync_rate_hz").value)
        self.reconnect_period_sec = float(
            self.get_parameter("reconnect_period_sec").value
        )
        self.connect_log_interval_sec = float(
            self.get_parameter("connect_log_interval_sec").value
        )
        self.field_cm_to_dashboard = float(
            self.get_parameter("field_cm_to_dashboard").value
        )
        self.send_all_field_detections = bool(
            self.get_parameter("send_all_field_detections").value
        )
        self.block_class_substrings = list(
            self.get_parameter("block_class_substrings").value
        )
        self.object_history_sec = float(
            self.get_parameter("object_history_sec").value
        )
        self.camera_offset_x_cm = float(
            self.get_parameter("camera_offset_x_cm").value
        )
        self.camera_offset_y_cm = float(
            self.get_parameter("camera_offset_y_cm").value
        )
        self.camera_offset_z_cm = float(
            self.get_parameter("camera_offset_z_cm").value
        )
        self.enable_robot_channel = bool(
            self.get_parameter("enable_robot_channel").value
        )
        self.enable_blocks_channel = bool(
            self.get_parameter("enable_blocks_channel").value
        )
        self.enable_camera_channel = bool(
            self.get_parameter("enable_camera_channel").value
        )
        self.camera_id = str(self.get_parameter("camera_id").value)
        self.camera_label = str(self.get_parameter("camera_label").value)
        self.camera_stream_fps = float(self.get_parameter("camera_stream_fps").value)
        self.camera_preview_url = str(self.get_parameter("camera_preview_url").value)

        self.dashboard_active = self.enable_dashboard and bool(self.bot_id)

        if self.sync_rate_hz <= 0.0:
            raise ValueError("sync_rate_hz must be greater than 0")

        if self.camera_stream_fps <= 0.0:
            raise ValueError("camera_stream_fps must be greater than 0")

        self.dashboard_host = resolve_host(self.vexai_host)

        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(String, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(
            String, self.detection_topic, self.detection_callback, 10
        )
        self.create_subscription(
            String,
            self.mapped_objects_topic,
            self.mapped_objects_callback,
            10,
        )
        self.create_subscription(
            String, self.action_ack_topic, self.action_ack_callback, 10
        )

        self.state_lock = threading.Lock()
        self.ws_lock = threading.Lock()

        self.latest_pose: Optional[dict[str, float]] = None
        self.latest_blocks: list[dict[str, Any]] = []
        self.block_history: list[dict[str, Any]] = []
        self.latest_bot_holding: list[str] = []
        self.last_detection_debug: dict[str, Any] = {}
        self.blocks_source = "none"
        self.last_mapped_objects_time = 0.0
        self.pose_initialized = False
        self.blocks_initialized = False

        self.robot_ws = None
        self.blocks_ws = None
        self.camera_ws = None
        self.robot_connected = False
        self.blocks_connected = False
        self.camera_connected = False
        self.last_connect_attempt_sec = 0.0
        self.sync_count = 0
        self.blocks_sync_count = 0
        self.last_blocks_sent_count = -1
        self.frame_count = 0
        self.sync_error_count = 0
        self.frame_error_count = 0
        self.last_error = ""
        self.connect_fail_count = 0
        self.last_connect_log_sec = 0.0
        self.last_reported_block_count = -1

        self.create_timer(1.0 / self.sync_rate_hz, self.sync_timer_callback)
        if self.dashboard_active:
            self.create_timer(
                self.reconnect_period_sec,
                self.reconnect_timer_callback,
            )
            if self.enable_camera_channel:
                self.create_timer(
                    1.0 / self.camera_stream_fps,
                    self.camera_timer_callback,
                )

        if not self.enable_dashboard:
            self.get_logger().info(
                "Dashboard sync disabled (enable_dashboard=false). Robot runs offline."
            )
        elif not self.bot_id:
            self.get_logger().info(
                "Dashboard sync disabled (bot_id empty). Robot runs offline."
            )
        else:
            self.get_logger().info(f"Dashboard host: {self.dashboard_host}")
            self.get_logger().info(f"Dashboard bot_id={self.bot_id}")
            self.get_logger().info(
                "Dashboard will connect when the server is reachable; "
                "robot stack continues if it is not."
            )
            if self.enable_camera_channel:
                self.get_logger().info(
                    "Camera stream: "
                    f"{self.camera_preview_url} -> /ws/video ({self.camera_id})"
                )
            self._connect_dashboard()

        self.get_logger().info(f"Odom topic: {self.odom_topic}")
        self.get_logger().info(f"Detection topic: {self.detection_topic}")
        self.get_logger().info(
            f"Mapped objects topic: {self.mapped_objects_topic}"
        )
        self.get_logger().info(f"Action ACK topic: {self.action_ack_topic}")
        self.get_logger().info(
            f"send_all_field_detections={self.send_all_field_detections} "
            f"block_class_substrings={self.block_class_substrings}"
        )

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                return True
            if normalized in ("0", "false", "no", "off", ""):
                return False
        return default

    @property
    def connected(self) -> bool:
        checks = []
        if self.enable_robot_channel:
            checks.append(self.robot_connected)
        if self.enable_blocks_channel:
            checks.append(self.blocks_connected)
        if self.enable_camera_channel:
            checks.append(self.camera_connected)
        return all(checks) if checks else False

    def odom_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid odom JSON: {exc}")
            return

        if not packet.get("valid", False):
            return

        pose = {
            "x": float(packet["x"]) * self.field_cm_to_dashboard,
            "y": float(packet["y"]) * self.field_cm_to_dashboard,
            "heading_deg": float(packet.get("heading", 0.0)),
        }

        with self.state_lock:
            self.latest_pose = pose

    def mapped_objects_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid mapped objects JSON: {exc}")
            return

        frame_blocks = self.blocks_from_mapped_objects(
            packet.get("objects", []),
        )
        now = time.time()

        with self.state_lock:
            for block in frame_blocks:
                self.block_history.append({**block, "time": now})

            cutoff = now - self.object_history_sec
            self.block_history = [
                block
                for block in self.block_history
                if float(block.get("time", 0.0)) >= cutoff
            ]
            self.latest_blocks = self.deduplicate_blocks(self.block_history)
            self.blocks_source = "mapped_objects"
            self.last_mapped_objects_time = now
            self.last_detection_debug = {
                "source": "mapped_objects",
                "object_count": len(packet.get("objects", [])),
                "frame_block_count": len(frame_blocks),
                "block_count": len(self.latest_blocks),
            }

        block_count = len(self.latest_blocks)
        if block_count != self.last_reported_block_count:
            self.last_reported_block_count = block_count
            self.get_logger().info(
                "Mapped objects received: "
                f"objects_in_msg={len(packet.get('objects', []))} "
                f"frame_blocks={len(frame_blocks)} "
                f"history_blocks={block_count} "
                f"sample={json.dumps(packet.get('objects', [])[:2])}"
            )

    def blocks_from_mapped_objects(
        self,
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []

        for obj in objects:
            x_field_cm = obj.get("x_field_cm", obj.get("x"))
            y_field_cm = obj.get("y_field_cm", obj.get("y"))
            if x_field_cm is None or y_field_cm is None:
                continue

            try:
                x_field = float(x_field_cm)
                y_field = float(y_field_cm)
                x = x_field * self.field_cm_to_dashboard
                y = y_field * self.field_cm_to_dashboard
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
            except (TypeError, ValueError):
                continue

            block_id = str(obj.get("id", "")).strip()
            if not block_id:
                class_name = str(obj.get("class_name", "object")).lower()
                block_id = f"{class_name}-{round(x_field)}-{round(y_field)}"

            class_name = obj.get("class_name")
            color = (
                self.block_color_label(class_name)
                if class_name is not None
                else obj.get("color", "unknown")
            )

            entry: dict[str, Any] = {
                "id": block_id,
                "x": x,
                "y": y,
                "color": color,
            }

            if class_name is not None:
                entry["class_name"] = class_name

            confidence = obj.get("confidence")
            if confidence is not None:
                entry["confidence"] = confidence

            entry["x_field_cm"] = round(x_field, 1)
            entry["y_field_cm"] = round(y_field, 1)

            blocks.append(entry)

        return blocks

    def detection_callback(self, msg: String):
        if time.time() - self.last_mapped_objects_time < self.object_history_sec:
            return

        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid detection JSON: {exc}")
            return

        robot_pose = self.extract_robot_pose(packet)
        frame_blocks, debug = self.build_dashboard_blocks(
            packet.get("detections", []),
            robot_pose=robot_pose,
        )
        now = time.time()

        with self.state_lock:
            for block in frame_blocks:
                self.block_history.append({**block, "time": now})

            cutoff = now - self.object_history_sec
            self.block_history = [
                block
                for block in self.block_history
                if float(block.get("time", 0.0)) >= cutoff
            ]
            self.latest_blocks = self.deduplicate_blocks(self.block_history)
            self.blocks_source = "detections"
            self.last_detection_debug = debug

        block_count = len(self.latest_blocks)
        if block_count != self.last_reported_block_count:
            self.last_reported_block_count = block_count
            self.get_logger().info(
                "Dashboard mapped blocks updated: "
                f"history={block_count} frame={len(frame_blocks)} "
                f"detections={debug.get('detection_count', 0)} "
                f"with_field={debug.get('with_field_count', 0)} "
                f"projected={debug.get('projected_count', 0)} "
                f"skipped={debug.get('skipped_count', 0)}"
            )

    def extract_robot_pose(
        self,
        packet: dict[str, Any],
    ) -> Optional[dict[str, float]]:
        if not bool(packet.get("robot_pose_valid", True)):
            return None

        try:
            return {
                "x": float(packet.get("robot_x", 0.0)),
                "y": float(packet.get("robot_y", 0.0)),
                "heading_deg": float(packet.get("robot_heading_deg", 0.0)),
            }
        except (TypeError, ValueError):
            return None

    def project_detection_to_field(
        self,
        detection: dict[str, Any],
        robot_pose: dict[str, float],
    ) -> Optional[tuple[float, float]]:
        x_cm = detection.get("x_cm")
        z_cm = detection.get("z_cm")
        if x_cm is None or z_cm is None:
            return None

        try:
            x_cam = float(x_cm)
            z_cam = float(z_cm)
            if not (math.isfinite(x_cam) and math.isfinite(z_cam)):
                return None
        except (TypeError, ValueError):
            return None

        local_right_cm = self.camera_offset_x_cm + x_cam
        local_forward_cm = self.camera_offset_y_cm + z_cam
        heading_rad = math.radians(robot_pose["heading_deg"])

        forward_x = math.sin(heading_rad)
        forward_y = math.cos(heading_rad)
        right_x = math.cos(heading_rad)
        right_y = -math.sin(heading_rad)

        x_field = (
            robot_pose["x"]
            + local_forward_cm * forward_x
            + local_right_cm * right_x
        )
        y_field = (
            robot_pose["y"]
            + local_forward_cm * forward_y
            + local_right_cm * right_y
        )

        if not (math.isfinite(x_field) and math.isfinite(y_field)):
            return None

        return x_field, y_field

    def build_dashboard_blocks(
        self,
        detections: list[dict[str, Any]],
        *,
        robot_pose: Optional[dict[str, float]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build block payloads from detections, matching perception map behavior."""
        blocks: list[dict[str, Any]] = []
        projected_count = 0
        with_field_count = 0
        skipped_count = 0

        for detection in detections:
            class_name = str(
                detection.get("class_name", detection.get("class", ""))
            ).lower()

            if not self.send_all_field_detections and not any(
                token in class_name for token in self.block_class_substrings
            ):
                skipped_count += 1
                continue

            x_field_cm = detection.get("x_field_cm")
            y_field_cm = detection.get("y_field_cm")

            if x_field_cm is None or y_field_cm is None:
                if robot_pose is not None:
                    projected = self.project_detection_to_field(
                        detection,
                        robot_pose,
                    )
                    if projected is not None:
                        x_field_cm, y_field_cm = projected
                        projected_count += 1
                    else:
                        skipped_count += 1
                        continue
                else:
                    skipped_count += 1
                    continue
            else:
                with_field_count += 1

            try:
                x_field = float(x_field_cm)
                y_field = float(y_field_cm)
                x = x_field * self.field_cm_to_dashboard
                y = y_field * self.field_cm_to_dashboard
                if not (math.isfinite(x) and math.isfinite(y)):
                    skipped_count += 1
                    continue
            except (TypeError, ValueError):
                skipped_count += 1
                continue

            blocks.append(
                {
                    "id": (
                        f"{class_name}-{round(x_field)}-"
                        f"{round(y_field)}"
                    ),
                    "x": x,
                    "y": y,
                    "color": self.block_color_label(class_name),
                    "class_name": class_name,
                }
            )

        debug = {
            "detection_count": len(detections),
            "with_field_count": with_field_count,
            "projected_count": projected_count,
            "skipped_count": skipped_count,
            "frame_block_count": len(blocks),
            "robot_pose_available": robot_pose is not None,
        }
        return blocks, debug

    @staticmethod
    def block_color_label(class_name: str) -> str:
        """Human-readable block type for the dashboard color field."""
        label = str(class_name).strip().lower().replace("_", " ")
        return label if label else "unknown"

    @staticmethod
    def deduplicate_blocks(
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()

        for block in blocks:
            block_id = str(block.get("id", ""))
            if not block_id or block_id in seen:
                continue
            seen.add(block_id)
            cleaned = {
                key: value
                for key, value in block.items()
                if key != "time"
            }
            deduped.append(cleaned)

        return deduped

    def action_ack_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid action ACK JSON: {exc}")
            return

        holding_blocks = packet.get("holding_blocks")
        if holding_blocks is None:
            return

        bot_holding = ["held-block"] if bool(holding_blocks) else []

        with self.state_lock:
            self.latest_bot_holding = bot_holding

    def reconnect_timer_callback(self):
        if not self.dashboard_active or self.connected:
            return
        self._connect_dashboard()

    def _current_state_snapshot(
        self,
    ) -> tuple[
        Optional[dict[str, float]],
        list[dict[str, Any]],
        list[str],
        bool,
        bool,
    ]:
        with self.state_lock:
            pose = dict(self.latest_pose) if self.latest_pose is not None else None
            blocks = list(self.latest_blocks)
            bot_holding = list(self.latest_bot_holding)
            pose_initialized = self.pose_initialized
            blocks_initialized = self.blocks_initialized
        return pose, blocks, bot_holding, pose_initialized, blocks_initialized

    def sync_timer_callback(self):
        pose, blocks, bot_holding, pose_initialized, blocks_initialized = (
            self._current_state_snapshot()
        )

        if not self.dashboard_active or not self.connected:
            self.publish_status(pose=pose, blocks=blocks, bot_holding=bot_holding)
            return

        try:
            with self.ws_lock:
                if (
                    self.enable_robot_channel
                    and pose is not None
                    and self.robot_ws is not None
                ):
                    send_location(
                        self.robot_ws,
                        x=pose["x"],
                        y=pose["y"],
                        heading_deg=pose["heading_deg"],
                        patch=pose_initialized,
                    )
                    with self.state_lock:
                        self.pose_initialized = True

                if self.enable_blocks_channel and self.blocks_ws is not None:
                    blocks_payload = {
                        "type": "patch" if blocks_initialized else "state",
                        "data": {
                            "blocks": blocks,
                            "bot_holding": bot_holding,
                        },
                    }
                    self.get_logger().info(
                        "WS /ws/blocks send "
                        f"bot_id={self.bot_id} "
                        f"blocks_source={self.blocks_source} "
                        f"{json.dumps(blocks_payload)}"
                    )
                    send_blocks(
                        self.blocks_ws,
                        blocks=blocks,
                        bot_holding=bot_holding,
                        patch=blocks_initialized,
                        wait_ack=False,
                    )
                    with self.state_lock:
                        self.blocks_initialized = True
                    self.blocks_sync_count += 1
                    self.last_blocks_sent_count = len(blocks)

            self.sync_count += 1
            self.last_error = ""
        except Exception as exc:
            self.sync_error_count += 1
            self.last_error = str(exc)
            self._log_dashboard_issue("sync", exc)
            self._disconnect_dashboard()

        self.publish_status(pose=pose, blocks=blocks, bot_holding=bot_holding)

    def camera_timer_callback(self):
        if not self.camera_connected or self.camera_ws is None:
            return

        frame = self._fetch_camera_frame()
        if frame is None:
            return

        try:
            with self.ws_lock:
                if self.camera_ws is not None:
                    send_frame(self.camera_ws, frame)
            self.frame_count += 1
            self.last_error = ""
        except Exception as exc:
            self.frame_error_count += 1
            self.last_error = str(exc)
            self._log_dashboard_issue("camera", exc)
            self.camera_connected = False
            self._close_socket("camera")

    def _log_dashboard_issue(self, channel: str, exc: Exception):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_connect_log_sec < self.connect_log_interval_sec:
            return

        self.last_connect_log_sec = now
        self.get_logger().info(
            f"Dashboard {channel} unavailable ({exc}). "
            "Continuing offline; will retry in the background."
        )

    def _fetch_camera_frame(self) -> Optional[bytes]:
        try:
            request = urllib.request.Request(
                self.camera_preview_url,
                headers={"Cache-Control": "no-cache"},
            )
            with urllib.request.urlopen(request, timeout=1.0) as response:
                if response.status != 200:
                    return None
                return response.read()
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None

    def _connect_channel(self, channel: str):
        if channel == "robot" and self.enable_robot_channel and not self.robot_connected:
            self.robot_ws, robot_hello = connect_robot(
                self.bot_id,
                host=self.vexai_host,
            )
            self.robot_connected = True
            self.get_logger().info(
                "Connected robot channel to VexAI dashboard "
                f"({robot_hello.get('type', 'ok')})"
            )
            return

        if channel == "blocks" and self.enable_blocks_channel and not self.blocks_connected:
            self.blocks_ws, blocks_hello = connect_blocks(
                self.bot_id,
                host=self.vexai_host,
            )
            self.blocks_connected = True
            self.get_logger().info(
                "Connected blocks channel to VexAI dashboard "
                f"({blocks_hello.get('type', 'ok')})"
            )
            return

        if (
            channel == "camera"
            and self.enable_camera_channel
            and not self.camera_connected
        ):
            self.camera_ws, camera_hello = connect_camera(
                self.bot_id,
                camera_id=self.camera_id,
                label=self.camera_label,
                host=self.vexai_host,
            )
            self.camera_connected = True
            self.get_logger().info(
                "Connected camera channel to VexAI dashboard "
                f"({camera_hello.get('type', 'ok')})"
            )

    def _connect_dashboard(self):
        if not self.dashboard_active:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_connect_attempt_sec < self.reconnect_period_sec:
            return

        self.last_connect_attempt_sec = now
        had_failure = False

        with self.ws_lock:
            for channel in ("robot", "blocks", "camera"):
                try:
                    self._connect_channel(channel)
                except Exception as exc:
                    had_failure = True
                    self.last_error = str(exc)
                    self.connect_fail_count += 1
                    self._close_socket(channel)

        if self.connected:
            self.connect_fail_count = 0
            self.last_error = ""
            with self.state_lock:
                self.pose_initialized = False
                self.blocks_initialized = False
            return

        if had_failure:
            self._log_dashboard_issue("connect", ConnectionError(self.last_error))

    def _close_socket(self, channel: str):
        socket_map = {
            "robot": "robot_ws",
            "blocks": "blocks_ws",
            "camera": "camera_ws",
        }
        attr = socket_map.get(channel)
        if attr is None:
            return

        ws = getattr(self, attr)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        setattr(self, attr, None)

        if channel == "robot":
            self.robot_connected = False
        elif channel == "blocks":
            self.blocks_connected = False
        elif channel == "camera":
            self.camera_connected = False

    def _disconnect_dashboard(self):
        with self.ws_lock:
            for channel in ("robot", "blocks", "camera"):
                self._close_socket(channel)

        with self.state_lock:
            self.pose_initialized = False
            self.blocks_initialized = False

    def publish_status(
        self,
        *,
        pose: Optional[dict[str, float]],
        blocks: list[dict[str, Any]],
        bot_holding: list[str],
    ):
        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "bot_id": self.bot_id,
            "dashboard_host": self.dashboard_host,
            "dashboard_active": self.dashboard_active,
            "offline_mode": self.dashboard_active and not self.connected,
            "connected": self.connected,
            "robot_connected": self.robot_connected,
            "blocks_connected": self.blocks_connected,
            "camera_connected": self.camera_connected,
            "sync_count": self.sync_count,
            "blocks_sync_count": self.blocks_sync_count,
            "last_blocks_sent_count": self.last_blocks_sent_count,
            "frame_count": self.frame_count,
            "sync_error_count": self.sync_error_count,
            "frame_error_count": self.frame_error_count,
            "last_error": self.last_error,
            "pose": pose,
            "block_count": len(blocks),
            "blocks": blocks,
            "blocks_source": self.blocks_source,
            "detection_debug": self.last_detection_debug,
            "bot_holding": bot_holding,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.status_pub.publish(msg)

    def destroy_node(self):
        self._disconnect_dashboard()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = DashboardNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Dashboard node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
