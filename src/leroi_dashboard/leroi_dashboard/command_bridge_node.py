import json
import queue
import threading
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from leroi_dashboard.leroi_dashboard import (
    connect_command,
    resolve_host,
    send_command_message,
)


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_CHASSIS_ONLY",
    2: "COLLECT_BLOCKS",
    3: "SCORE_LEFT_LONG_GOAL_ALLIANCE",
    4: "SCORE_LEFT_LONG_GOAL_OPPONENT",
    5: "SCORE_RIGHT_LONG_GOAL_ALLIANCE",
    6: "SCORE_RIGHT_LONG_GOAL_OPPONENT",
    7: "SCORE_MID_LEFT",
    8: "SCORE_MID_RIGHT",
    9: "SCORE_LOW_LEFT",
    10: "SCORE_LOW_RIGHT",
    11: "DESCORE_LEFT_LONG_GOAL_ALLIANCE_FIELD",
    12: "DESCORE_LEFT_LONG_GOAL_ALLIANCE_WALL",
    13: "DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD",
    14: "DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL",
    15: "DESCORE_RIGHT_LONG_GOAL_ALLIANCE_FIELD",
    16: "DESCORE_RIGHT_LONG_GOAL_ALLIANCE_WALL",
    17: "DESCORE_RIGHT_LONG_GOAL_OPPONENT_FIIELD",
    18: "DESCORE_RIGHT_LONG_GOAL_OPPONENT_WALL",
    19: "RAM_RIGHT_LONG_GOAL_OPPONENT",
    20: "RAM_RIGHT_LONG_GOAL_ALLIANCE",
}


class CommandBridgeNode(Node):
    def __init__(self):
        super().__init__("command_bridge_node")

        self.declare_parameter("enable_command_bridge", True)
        self.declare_parameter("bot_id", "")
        self.declare_parameter("vexai_host", "")
        self.declare_parameter("action_request_topic", "/decision/action_request")
        self.declare_parameter("action_ack_topic", "/robot/action_ack")
        self.declare_parameter("rejected_topic", "/decision/rejected")
        self.declare_parameter("command_log_topic", "/decision/command_log")
        self.declare_parameter("decision_status_topic", "/decision/status")
        self.declare_parameter("status_topic", "/command_bridge/status")
        self.declare_parameter("status_rate_hz", 2.0)
        self.declare_parameter("reconnect_period_sec", 5.0)
        self.declare_parameter("connect_log_interval_sec", 60.0)
        self.declare_parameter("ws_recv_timeout_sec", 1.0)

        self.enable_command_bridge = self._parse_bool(
            self.get_parameter("enable_command_bridge").value,
            default=True,
        )
        self.bot_id = str(self.get_parameter("bot_id").value).strip()
        self.vexai_host = str(self.get_parameter("vexai_host").value).strip() or None
        self.action_request_topic = str(
            self.get_parameter("action_request_topic").value
        )
        self.action_ack_topic = str(self.get_parameter("action_ack_topic").value)
        self.rejected_topic = str(self.get_parameter("rejected_topic").value)
        self.command_log_topic = str(self.get_parameter("command_log_topic").value)
        self.decision_status_topic = str(
            self.get_parameter("decision_status_topic").value
        )
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_rate_hz = float(self.get_parameter("status_rate_hz").value)
        self.reconnect_period_sec = float(
            self.get_parameter("reconnect_period_sec").value
        )
        self.connect_log_interval_sec = float(
            self.get_parameter("connect_log_interval_sec").value
        )
        self.ws_recv_timeout_sec = float(
            self.get_parameter("ws_recv_timeout_sec").value
        )

        if self.status_rate_hz <= 0.0:
            raise ValueError("status_rate_hz must be greater than 0")

        self.bridge_active = self.enable_command_bridge and bool(self.bot_id)
        self.dashboard_host = resolve_host(self.vexai_host)

        self.action_request_pub = self.create_publisher(
            String,
            self.action_request_topic,
            10,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(
            String,
            self.action_ack_topic,
            self.action_ack_callback,
            10,
        )
        self.create_subscription(
            String,
            self.rejected_topic,
            self.rejected_callback,
            10,
        )
        self.create_subscription(
            String,
            self.command_log_topic,
            self.command_log_callback,
            10,
        )
        self.create_subscription(
            String,
            self.decision_status_topic,
            self.decision_status_callback,
            10,
        )

        self.state_lock = threading.Lock()
        self.ws_lock = threading.Lock()
        self.incoming_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        self.pending_by_request_id: dict[str, dict[str, Any]] = {}
        self.request_id_by_command_id: dict[int, str] = {}
        self.latest_decision_status: Optional[dict[str, Any]] = None

        self.command_ws = None
        self.command_connected = False
        self.recv_thread: Optional[threading.Thread] = None
        self.recv_thread_running = False
        self.last_connect_attempt_sec = 0.0
        self.last_connect_log_sec = 0.0
        self.last_error = ""
        self.received_count = 0
        self.forwarded_count = 0
        self.rejected_count = 0
        self.result_count = 0

        self.create_timer(1.0 / self.status_rate_hz, self.status_timer_callback)
        self.create_timer(0.05, self.process_incoming_timer_callback)

        if self.bridge_active:
            self.create_timer(
                self.reconnect_period_sec,
                self.reconnect_timer_callback,
            )
            self._connect_command_channel()

        self.get_logger().info("Command bridge node started.")
        self.get_logger().info(f"Bridge active: {self.bridge_active}")
        if self.bridge_active:
            self.get_logger().info(f"Command host: {self.dashboard_host}")
            self.get_logger().info(f"Command bot_id={self.bot_id}")
        self.get_logger().info(
            f"Action request topic: {self.action_request_topic}"
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

    def reconnect_timer_callback(self):
        if not self.bridge_active or self.command_connected:
            return
        self._connect_command_channel()

    def process_incoming_timer_callback(self):
        while True:
            try:
                packet = self.incoming_queue.get_nowait()
            except queue.Empty:
                return
            self.handle_ws_packet(packet)

    def status_timer_callback(self):
        if (
            self.bridge_active
            and self.command_connected
            and self.latest_decision_status is not None
        ):
            self._send_ws(self.build_command_status_packet())

        self.publish_local_status()

    def decision_status_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if isinstance(packet, dict):
            self.latest_decision_status = packet

    def command_log_callback(self, msg: String):
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if event.get("event") != "COMMAND_SENT":
            return

        command = event.get("command") or {}
        params = command.get("params") or {}
        request_id = params.get("request_id")

        if not request_id:
            return

        with self.state_lock:
            pending = self.pending_by_request_id.get(request_id)

        if pending is None:
            return

        command_id = int(event["id"])
        action_id = int(event["action_id"])
        action_name = str(event.get("action_name", ACTION_NAMES.get(action_id, "")))

        with self.state_lock:
            pending["command_id"] = command_id
            self.request_id_by_command_id[command_id] = request_id

        self._send_ws(
            {
                "type": "action_ack",
                "request_id": request_id,
                "status": "QUEUED",
                "command_id": command_id,
                "action_id": action_id,
                "action_name": action_name,
                "message": "Approved by decision layer",
            }
        )

        self.get_logger().info(
            f"Forwarded action request_id={request_id} "
            f"command_id={command_id} action_id={action_id}"
        )

    def rejected_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        request_id = self.extract_request_id(packet)
        if not request_id:
            return

        command = packet.get("command") or {}
        action_id = command.get("action_id")

        with self.state_lock:
            self.pending_by_request_id.pop(request_id, None)
            command_id = command.get("id")
            if command_id is not None:
                self.request_id_by_command_id.pop(int(command_id), None)

        self._send_ws(
            {
                "type": "action_ack",
                "request_id": request_id,
                "status": "REJECTED",
                "reason": packet.get("reason", "REJECTED"),
                "action_id": action_id,
            }
        )
        self.rejected_count += 1

        self.get_logger().warning(
            f"Rejected action request_id={request_id} "
            f"reason={packet.get('reason', 'REJECTED')}"
        )

    def action_ack_callback(self, msg: String):
        try:
            ack = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        command_id = self.safe_int(ack.get("command_id"), default=-1)
        if command_id < 0:
            return

        with self.state_lock:
            request_id = self.request_id_by_command_id.get(command_id)

        if not request_id:
            return

        action_id = self.safe_int(ack.get("action_id"), default=-1)
        status = str(ack.get("status", "UNKNOWN")).upper()
        action_name = ACTION_NAMES.get(action_id, f"ACTION_{action_id}")

        self._send_ws(
            {
                "type": "action_result",
                "request_id": request_id,
                "command_id": command_id,
                "action_id": action_id,
                "action_name": action_name,
                "status": status,
                "holding_blocks": ack.get("holding_blocks"),
                "detail": ack.get("detail", ""),
            }
        )
        self.result_count += 1

        with self.state_lock:
            self.pending_by_request_id.pop(request_id, None)
            self.request_id_by_command_id.pop(command_id, None)

        self.get_logger().info(
            f"Action result request_id={request_id} "
            f"command_id={command_id} status={status}"
        )

    def handle_ws_packet(self, packet: dict[str, Any]):
        message_type = str(packet.get("type", "")).lower()

        if message_type in ("hello", "ping"):
            return

        if message_type != "action_request":
            self.get_logger().info(
                f"Ignoring unsupported WS message type={packet.get('type')}"
            )
            return

        try:
            request_id = str(packet["request_id"]).strip()
            action_id = int(packet["action_id"])
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().warning(f"Invalid action_request payload: {exc}")
            return

        if not request_id:
            self.get_logger().warning("action_request missing request_id")
            return

        if action_id < 0 or action_id > 20:
            self._send_ws(
                {
                    "type": "action_ack",
                    "request_id": request_id,
                    "status": "REJECTED",
                    "reason": "INVALID_ACTION_ID",
                    "action_id": action_id,
                }
            )
            return

        ros_payload = self.build_ros_action_request(packet, request_id, action_id)

        with self.state_lock:
            self.pending_by_request_id[request_id] = {
                "action_id": action_id,
                "command_id": packet.get("command_id"),
            }

        out = String()
        out.data = json.dumps(ros_payload)
        self.action_request_pub.publish(out)
        self.forwarded_count += 1

        self.get_logger().info(
            f"Received WS action request_id={request_id} action_id={action_id}"
        )

    def build_ros_action_request(
        self,
        packet: dict[str, Any],
        request_id: str,
        action_id: int,
    ) -> dict[str, Any]:
        params = dict(packet.get("params") or {})
        params["request_id"] = request_id

        ros_payload: dict[str, Any] = {
            "type": "ACTION",
            "action_id": action_id,
            "points": packet.get("points") or [],
            "params": params,
        }

        if "command_id" in packet and packet["command_id"] is not None:
            ros_payload["id"] = int(packet["command_id"])

        return ros_payload

    def build_command_status_packet(self) -> dict[str, Any]:
        status = self.latest_decision_status or {}
        return {
            "type": "command_status",
            "bot_id": self.bot_id,
            "waiting_for_ack": status.get("waiting_for_ack", False),
            "pending_command_id": status.get("pending_command_id"),
            "pending_action_id": status.get("pending_action_id"),
            "pending_action_name": status.get("pending_action_name"),
            "blocks_collected": status.get("blocks_collected", False),
            "robot_has_blocks": status.get("robot_has_blocks", False),
            "last_ack": status.get("last_ack"),
        }

    def extract_request_id(self, packet: dict[str, Any]) -> Optional[str]:
        command = packet.get("command")
        if isinstance(command, dict):
            params = command.get("params") or {}
            request_id = params.get("request_id")
            if request_id:
                return str(request_id)

        raw = packet.get("raw")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None

            if isinstance(parsed, dict):
                params = parsed.get("params") or {}
                request_id = params.get("request_id")
                if request_id:
                    return str(request_id)

        return None

    def _connect_command_channel(self):
        if not self.bridge_active:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_connect_attempt_sec < self.reconnect_period_sec:
            return

        self.last_connect_attempt_sec = now

        try:
            with self.ws_lock:
                self._close_command_socket()
                self.command_ws, hello = connect_command(
                    self.bot_id,
                    host=self.vexai_host,
                )
                self.command_connected = True
                self.last_error = ""
                self._start_recv_thread()

            self.get_logger().info(
                "Connected command channel to VexAI dashboard "
                f"({hello.get('type', 'ok')})"
            )
        except Exception as exc:
            self.command_connected = False
            self.last_error = str(exc)
            self._close_command_socket()
            self._log_connect_issue(exc)

    def _start_recv_thread(self):
        self._stop_recv_thread()
        self.recv_thread_running = True
        self.recv_thread = threading.Thread(
            target=self._recv_loop,
            name="command_ws_recv",
            daemon=True,
        )
        self.recv_thread.start()

    def _stop_recv_thread(self):
        self.recv_thread_running = False
        if self.recv_thread is not None and self.recv_thread.is_alive():
            self.recv_thread.join(timeout=2.0)
        self.recv_thread = None

    def _recv_loop(self):
        while self.recv_thread_running and self.command_connected:
            ws = self.command_ws
            if ws is None:
                break

            try:
                raw = ws.recv(timeout=self.ws_recv_timeout_sec)
            except TimeoutError:
                continue
            except Exception as exc:
                self.last_error = str(exc)
                break

            try:
                packet = json.loads(raw)
            except json.JSONDecodeError as exc:
                self.get_logger().warning(f"Invalid command WS JSON: {exc}")
                continue

            if not isinstance(packet, dict):
                continue

            self.received_count += 1
            self.incoming_queue.put(packet)

        self.command_connected = False
        self._close_command_socket()

    def _send_ws(self, payload: dict[str, Any]):
        if not self.command_connected or self.command_ws is None:
            return

        try:
            with self.ws_lock:
                if self.command_ws is not None:
                    send_command_message(self.command_ws, payload)
        except Exception as exc:
            self.last_error = str(exc)
            self.command_connected = False
            self._close_command_socket()

    def _close_command_socket(self):
        self._stop_recv_thread()

        with self.ws_lock:
            if self.command_ws is not None:
                try:
                    self.command_ws.close()
                except Exception:
                    pass
                self.command_ws = None

        self.command_connected = False

    def _log_connect_issue(self, exc: Exception):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_connect_log_sec < self.connect_log_interval_sec:
            return

        self.last_connect_log_sec = now
        self.get_logger().info(
            f"Command channel unavailable ({exc}). "
            "Continuing offline; will retry in the background."
        )

    def publish_local_status(self):
        with self.state_lock:
            pending_count = len(self.pending_by_request_id)

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "bot_id": self.bot_id,
            "bridge_active": self.bridge_active,
            "command_connected": self.command_connected,
            "dashboard_host": self.dashboard_host,
            "received_count": self.received_count,
            "forwarded_count": self.forwarded_count,
            "rejected_count": self.rejected_count,
            "result_count": self.result_count,
            "pending_count": pending_count,
            "last_error": self.last_error,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.status_pub.publish(msg)

    def destroy_node(self):
        self._close_command_socket()
        super().destroy_node()

    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default


def main(args=None):
    rclpy.init(args=args)

    node = CommandBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Command bridge node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
