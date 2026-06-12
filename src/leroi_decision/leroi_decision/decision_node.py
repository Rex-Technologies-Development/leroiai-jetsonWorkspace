import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


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

    17: "DESCORE_RIGHT_LONG_GOAL_OPPONENT_FIELD",
    18: "DESCORE_RIGHT_LONG_GOAL_OPPONENT_WALL",

    19: "RAM_RIGHT_LONG_GOAL_OPPONENT",
    20: "RAM_RIGHT_LONG_GOAL_ALLIANCE",

    21: "RAM_LEFT_LONG_GOAL_OPPONENT",
    22: "RAM_LEFT_LONG_GOAL_ALLIANCE",
    24: "RESET_GYRO",
}

STOP_ACTION_ID = 0
COLLECT_BLOCKS_ACTION_ID = 2
RESET_GYRO_ACTION_ID = 24
SCORE_ACTION_IDS = set(range(3, 11))

# The full defined action space. Commands with any other action_id are rejected
# so an undefined command can never reach the serial messenger / V5.
KNOWN_ACTION_IDS = set(ACTION_NAMES)


class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")

        self.declare_parameter("action_request_topic", "/decision/action_request")
        self.declare_parameter("robot_command_topic", "/robot/command")
        self.declare_parameter("action_ack_topic", "/robot/action_ack")
        self.declare_parameter("status_topic", "/decision/status")
        self.declare_parameter("rejected_topic", "/decision/rejected")
        self.declare_parameter("command_log_topic", "/decision/command_log")

        self.declare_parameter(
            "command_log_path",
            "~/leroiai-jetsonWorkspace/logs/decision_commands.jsonl",
        )
        self.declare_parameter("command_history_max", 100)

        self.declare_parameter("status_rate_hz", 2.0)
        self.declare_parameter("command_timeout_sec", 15.0)

        self.declare_parameter("allow_score_without_blocks", False)
        self.declare_parameter("block_new_action_while_waiting", True)
        self.declare_parameter("allow_stop_while_waiting", True)
        # Reject any action_id that is not in the defined action space.
        self.declare_parameter("reject_unknown_action_ids", True)
        # RESET_GYRO must be able to interrupt an in-flight action (robot stops,
        # stands still, ACKs) so we can recalibrate the heading mid-routine.
        self.declare_parameter("allow_gyro_reset_while_waiting", True)

        self.action_request_topic = str(
            self.get_parameter("action_request_topic").value
        )
        self.robot_command_topic = str(
            self.get_parameter("robot_command_topic").value
        )
        self.action_ack_topic = str(self.get_parameter("action_ack_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.rejected_topic = str(self.get_parameter("rejected_topic").value)
        self.command_log_topic = str(self.get_parameter("command_log_topic").value)

        self.command_log_path = os.path.expanduser(
            str(self.get_parameter("command_log_path").value)
        )
        self.command_history_max = int(
            self.get_parameter("command_history_max").value
        )

        self.status_rate_hz = float(self.get_parameter("status_rate_hz").value)
        self.command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )

        self.allow_score_without_blocks = bool(
            self.get_parameter("allow_score_without_blocks").value
        )
        self.block_new_action_while_waiting = bool(
            self.get_parameter("block_new_action_while_waiting").value
        )
        self.allow_stop_while_waiting = bool(
            self.get_parameter("allow_stop_while_waiting").value
        )
        self.allow_gyro_reset_while_waiting = bool(
            self.get_parameter("allow_gyro_reset_while_waiting").value
        )
        self.reject_unknown_action_ids = bool(
            self.get_parameter("reject_unknown_action_ids").value
        )

        self.command_pub = self.create_publisher(
            String,
            self.robot_command_topic,
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10,
        )

        self.rejected_pub = self.create_publisher(
            String,
            self.rejected_topic,
            10,
        )

        self.command_log_pub = self.create_publisher(
            String,
            self.command_log_topic,
            10,
        )

        self.action_request_sub = self.create_subscription(
            String,
            self.action_request_topic,
            self.action_request_callback,
            10,
        )

        self.action_ack_sub = self.create_subscription(
            String,
            self.action_ack_topic,
            self.action_ack_callback,
            10,
        )

        self.status_timer = self.create_timer(
            1.0 / self.status_rate_hz,
            self.status_timer_callback,
        )

        self.next_command_id = 1

        # True after COLLECT_BLOCKS completes; cleared when a score command is sent.
        self.blocks_collected = False
        self.robot_has_blocks = False

        self.waiting_for_ack = False
        self.pending_command_id: Optional[int] = None
        self.pending_action_id: Optional[int] = None
        self.pending_action_name: Optional[str] = None
        self.pending_sent_time: Optional[float] = None

        self.last_ack: Optional[Dict[str, Any]] = None
        self.last_rejection: Optional[Dict[str, Any]] = None
        self.command_history: List[Dict[str, Any]] = []

        if self.command_log_path:
            log_dir = os.path.dirname(self.command_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        self.get_logger().info("Decision node started.")
        self.get_logger().info(f"Action request topic: {self.action_request_topic}")
        self.get_logger().info(f"Robot command topic: {self.robot_command_topic}")
        self.get_logger().info(f"Action ACK topic: {self.action_ack_topic}")
        self.get_logger().info(f"Command log topic: {self.command_log_topic}")
        self.get_logger().info(f"Command log path: {self.command_log_path}")
        self.get_logger().info(
            f"allow_score_without_blocks={self.allow_score_without_blocks}"
        )

    def allocate_command_id(self) -> int:
        command_id = self.next_command_id
        self.next_command_id += 1
        return command_id

    def action_request_callback(self, msg: String):
        try:
            command = self.normalize_action_request(msg.data)
        except ValueError as exc:
            self.reject_command(
                raw=msg.data,
                reason=f"INVALID_COMMAND: {exc}",
            )
            return

        allowed, reason = self.is_command_allowed(command)

        if not allowed:
            self.reject_command(
                raw=msg.data,
                reason=reason,
                command=command,
            )
            return

        action_id = int(command["action_id"])

        out = String()
        out.data = json.dumps(command)
        self.command_pub.publish(out)

        self.save_sent_command(command)

        # Note: blocks_collected is NOT cleared here. The robot is still holding
        # blocks while it drives to score; we clear it when the score ACK returns
        # (see action_ack_callback).

        self.waiting_for_ack = True
        self.pending_command_id = int(command["id"])
        self.pending_action_id = action_id
        self.pending_action_name = str(command["action_name"])
        self.pending_sent_time = time.time()

        self.get_logger().info(
            f"Approved command id={command['id']} "
            f"action_id={action_id} "
            f"action={command['action_name']} "
            f"points={len(command.get('points', []))} "
            f"blocks_collected={self.blocks_collected}"
        )

    def normalize_action_request(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()

        if not text:
            raise ValueError("empty command")

        try:
            action_id = int(text)
            data: Dict[str, Any] = {
                "type": "ACTION",
                "id": self.allocate_command_id(),
                "action_id": action_id,
                "points": [],
                "params": {},
            }
        except ValueError:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"not an integer or JSON: {exc}") from exc

            if not isinstance(parsed, dict):
                raise ValueError("JSON command must be an object")

            data = parsed

        if "id" not in data:
            data["id"] = self.allocate_command_id()

        try:
            data["id"] = int(data["id"])
        except (ValueError, TypeError) as exc:
            raise ValueError("id must be an integer") from exc

        if data["id"] >= self.next_command_id:
            self.next_command_id = data["id"] + 1

        if "action_id" not in data:
            raise ValueError("missing action_id")

        try:
            data["action_id"] = int(data["action_id"])
        except (ValueError, TypeError) as exc:
            raise ValueError("action_id must be an integer") from exc

        action_id = int(data["action_id"])

        data["type"] = "ACTION"
        data["action_name"] = ACTION_NAMES.get(action_id, f"ACTION_{action_id}")

        points = data.get("points", [])
        if points is None:
            points = []

        data["points"] = self.normalize_points(points)

        params = data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")

        data["params"] = params

        return data

    def normalize_points(self, points: Any) -> List[Dict[str, float]]:
        if not isinstance(points, list):
            raise ValueError("points must be a list")

        normalized = []

        for i, point in enumerate(points):
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
                    raise ValueError("point must be dict or list")

                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and math.isfinite(heading)
                ):
                    raise ValueError("point values must be finite")

                normalized.append(
                    {
                        "x": x,
                        "y": y,
                        "heading": heading,
                    }
                )

            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raise ValueError(f"invalid point at index {i}: {exc}") from exc

        return normalized

    def is_command_allowed(self, command: Dict[str, Any]):
        action_id = int(command["action_id"])

        if self.reject_unknown_action_ids and action_id not in KNOWN_ACTION_IDS:
            return (
                False,
                f"UNKNOWN_ACTION_ID: {action_id} is not in the defined action space",
            )

        if self.waiting_for_ack:
            interrupt_allowed = (
                self.allow_stop_while_waiting and action_id == STOP_ACTION_ID
            ) or (
                self.allow_gyro_reset_while_waiting
                and action_id == RESET_GYRO_ACTION_ID
            )

            if self.block_new_action_while_waiting and not interrupt_allowed:
                return (
                    False,
                    f"BUSY_WAITING_FOR_ACK: pending_command_id={self.pending_command_id}",
                )

        if action_id in SCORE_ACTION_IDS:
            if not self.blocks_collected and not self.allow_score_without_blocks:
                return (
                    False,
                    "REJECTED_NO_BLOCKS: scoring requires successful COLLECT_BLOCKS first",
                )

        return True, "OK"

    def set_blocks_collected(self, collected: bool):
        self.blocks_collected = collected
        self.robot_has_blocks = collected

    def clear_blocks_collected(self):
        if self.blocks_collected:
            self.get_logger().info(
                "blocks_collected cleared: score complete, collect again before scoring"
            )
        self.set_blocks_collected(False)

    def save_sent_command(self, command: Dict[str, Any]):
        now_unix = time.time()
        now_ns = self.get_clock().now().nanoseconds

        event = {
            "event": "COMMAND_SENT",
            "sent_stamp_ns": now_ns,
            "sent_time_unix": now_unix,
            "id": int(command["id"]),
            "action_id": int(command["action_id"]),
            "action_name": str(command["action_name"]),
            "points": command.get("points", []),
            "params": command.get("params", {}),
            "command": command,
        }

        self.command_history.append(event)

        if len(self.command_history) > self.command_history_max:
            self.command_history = self.command_history[-self.command_history_max :]

        msg = String()
        msg.data = json.dumps(event)
        self.command_log_pub.publish(msg)

        if self.command_log_path:
            try:
                with open(self.command_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except OSError as exc:
                self.get_logger().warning(f"Failed to write command log: {exc}")

    def reject_command(
        self,
        raw: str,
        reason: str,
        command: Optional[Dict[str, Any]] = None,
    ):
        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "reason": reason,
            "raw": raw,
            "command": command,
            "blocks_collected": self.blocks_collected,
            "robot_has_blocks": self.robot_has_blocks,
            "waiting_for_ack": self.waiting_for_ack,
            "pending_command_id": self.pending_command_id,
            "pending_action_id": self.pending_action_id,
        }

        self.last_rejection = packet

        msg = String()
        msg.data = json.dumps(packet)
        self.rejected_pub.publish(msg)

        self.get_logger().warning(f"Rejected command: {reason}")

    def action_ack_callback(self, msg: String):
        try:
            ack = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid ACK JSON received.")
            return

        command_id = self.safe_int(ack.get("command_id"), default=-1)
        status = str(ack.get("status", "UNKNOWN")).upper()

        action_id_raw = ack.get("action_id")

        if action_id_raw is None and command_id == self.pending_command_id:
            action_id_raw = self.pending_action_id

        action_id = self.safe_int(action_id_raw, default=-1)

        holding_blocks = ack.get("holding_blocks")

        self.last_ack = ack

        if command_id == self.pending_command_id:
            completed_action_id = action_id

            self.waiting_for_ack = False
            self.pending_command_id = None
            self.pending_action_id = None
            self.pending_action_name = None
            self.pending_sent_time = None

            # We get no separate success/failure signal from the robot, so an ACK
            # for the in-flight command means that action is finished.
            if completed_action_id == COLLECT_BLOCKS_ACTION_ID:
                # Collect finished -> robot is now holding blocks, scoring allowed.
                self.set_blocks_collected(True)
                self.get_logger().info(
                    "blocks_collected set: collect complete, scoring allowed"
                )
            elif completed_action_id in SCORE_ACTION_IDS:
                # Score finished -> robot is now empty, must collect again first.
                self.clear_blocks_collected()

        # If the robot explicitly reports its hopper state, that is authoritative
        # and overrides whatever we inferred above.
        if holding_blocks is not None:
            self.set_blocks_collected(bool(holding_blocks))

        self.get_logger().info(
            f"ACK command_id={command_id} status={status} "
            f"action_id={action_id} blocks_collected={self.blocks_collected}"
        )

    def status_timer_callback(self):
        now = time.time()

        if self.waiting_for_ack and self.pending_sent_time is not None:
            age = now - self.pending_sent_time

            if age > self.command_timeout_sec:
                self.get_logger().warning(
                    f"Command timeout: id={self.pending_command_id} "
                    f"action_id={self.pending_action_id}"
                )

                self.waiting_for_ack = False
                self.pending_command_id = None
                self.pending_action_id = None
                self.pending_action_name = None
                self.pending_sent_time = None

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "blocks_collected": self.blocks_collected,
            "needs_collect_blocks": not self.blocks_collected,
            "robot_has_blocks": self.robot_has_blocks,
            "waiting_for_ack": self.waiting_for_ack,
            "pending_command_id": self.pending_command_id,
            "pending_action_id": self.pending_action_id,
            "pending_action_name": self.pending_action_name,
            "last_ack": self.last_ack,
            "last_rejection": self.last_rejection,
            "last_sent_command": self.command_history[-1]
            if self.command_history
            else None,
            "command_history_size": len(self.command_history),
            "command_log_path": self.command_log_path,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.status_pub.publish(msg)

    @staticmethod
    def safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default


def main(args=None):
    rclpy.init(args=args)

    node = DecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Decision node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()