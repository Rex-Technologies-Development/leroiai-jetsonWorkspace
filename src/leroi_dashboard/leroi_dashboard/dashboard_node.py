import json
import threading
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
        self.declare_parameter("action_ack_topic", "/robot/action_ack")
        self.declare_parameter("status_topic", "/dashboard/status")
        self.declare_parameter("sync_rate_hz", 2.0)
        self.declare_parameter("reconnect_period_sec", 5.0)
        self.declare_parameter("connect_log_interval_sec", 60.0)
        self.declare_parameter("field_cm_to_dashboard", 1.0 / 2.54)
        self.declare_parameter("block_class_substrings", ["block"])
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
        self.block_class_substrings = list(
            self.get_parameter("block_class_substrings").value
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
            String, self.action_ack_topic, self.action_ack_callback, 10
        )

        self.state_lock = threading.Lock()
        self.ws_lock = threading.Lock()

        self.latest_pose: Optional[dict[str, float]] = None
        self.latest_blocks: list[dict[str, Any]] = []
        self.latest_bot_holding: list[str] = []
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
        self.frame_count = 0
        self.sync_error_count = 0
        self.frame_error_count = 0
        self.last_error = ""
        self.connect_fail_count = 0
        self.last_connect_log_sec = 0.0

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
        self.get_logger().info(f"Action ACK topic: {self.action_ack_topic}")

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

    def detection_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid detection JSON: {exc}")
            return

        blocks = []
        for index, detection in enumerate(packet.get("detections", [])):
            class_name = str(
                detection.get("class_name", detection.get("class", ""))
            ).lower()

            if not any(token in class_name for token in self.block_class_substrings):
                continue

            x_field_cm = detection.get("x_field_cm")
            y_field_cm = detection.get("y_field_cm")
            if x_field_cm is None or y_field_cm is None:
                continue

            color = "unknown"
            if "blue" in class_name:
                color = "blue"
            elif "red" in class_name:
                color = "red"

            blocks.append(
                {
                    "id": f"block-{index}",
                    "x": float(x_field_cm) * self.field_cm_to_dashboard,
                    "y": float(y_field_cm) * self.field_cm_to_dashboard,
                    "color": color,
                }
            )

        with self.state_lock:
            self.latest_blocks = blocks

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
                    send_blocks(
                        self.blocks_ws,
                        blocks=blocks,
                        bot_holding=bot_holding,
                        patch=blocks_initialized,
                    )
                    with self.state_lock:
                        self.blocks_initialized = True

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
            "frame_count": self.frame_count,
            "sync_error_count": self.sync_error_count,
            "frame_error_count": self.frame_error_count,
            "last_error": self.last_error,
            "pose": pose,
            "block_count": len(blocks),
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
