import json
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Default scenario if no sequence_json parameter is provided.
#
# 1) MOVE_CHASSIS_ONLY (action_id 1) to a first target point.
# 2) MOVE_CHASSIS_ONLY (action_id 1) to a second target point.
# 3) COLLECT_BLOCKS (action_id 2) with a bunch of target points.
#
# Field is 365.76 cm x 365.76 cm, origin bottom-left.
DEFAULT_SEQUENCE: List[Dict[str, Any]] = [
    {
        "action_id": 1,
        "points": [{"x": 120.0, "y": 90.0, "heading": 0.0}],
        "params": {"label": "move_to_point_1"},
    },
    {
        "action_id": 1,
        "points": [{"x": 240.0, "y": 180.0, "heading": 90.0}],
        "params": {"label": "move_to_point_2"},
    },
    {
        "action_id": 2,
        "points": [
            {"x": 250.0, "y": 200.0, "heading": 90.0},
            {"x": 270.0, "y": 220.0, "heading": 45.0},
            {"x": 290.0, "y": 240.0, "heading": 0.0},
            {"x": 300.0, "y": 260.0, "heading": -45.0},
        ],
        "params": {"label": "collect_blocks_run"},
    },
]


class DecisionTestDriver(Node):
    """Drives the decision node through a command sequence with ACK handshaking.

    Flow per command:
      1. Publish a command to /decision/action_request.
      2. The decision node approves it and forwards it to /robot/command.
      3. We observe /robot/command and (optionally) simulate the robot by
         publishing an ACK on /robot/action_ack after a short delay.
      4. When a matching DONE ACK is seen on /robot/action_ack, we advance and
         send the next command.

    Set simulate_robot_ack=False when a real robot/serial bridge is connected
    and will produce its own ACKs; the driver will still advance off those.
    """

    def __init__(self):
        super().__init__("decision_test_driver")

        self.declare_parameter("action_request_topic", "/decision/action_request")
        self.declare_parameter("robot_command_topic", "/robot/command")
        self.declare_parameter("action_ack_topic", "/robot/action_ack")

        self.declare_parameter("startup_delay_sec", 5.0)
        self.declare_parameter("sim_ack_delay_sec", 1.0)
        self.declare_parameter("inter_command_delay_sec", 1.0)
        self.declare_parameter("simulate_robot_ack", True)
        self.declare_parameter("loop_sequence", False)

        # JSON array of commands. Empty string / "[]" -> DEFAULT_SEQUENCE.
        self.declare_parameter("sequence_json", "")

        self.action_request_topic = str(
            self.get_parameter("action_request_topic").value
        )
        self.robot_command_topic = str(
            self.get_parameter("robot_command_topic").value
        )
        self.action_ack_topic = str(self.get_parameter("action_ack_topic").value)

        self.startup_delay_sec = float(
            self.get_parameter("startup_delay_sec").value
        )
        self.sim_ack_delay_sec = float(self.get_parameter("sim_ack_delay_sec").value)
        self.inter_command_delay_sec = float(
            self.get_parameter("inter_command_delay_sec").value
        )
        self.simulate_robot_ack = bool(
            self.get_parameter("simulate_robot_ack").value
        )
        self.loop_sequence = bool(self.get_parameter("loop_sequence").value)

        self.sequence = self._load_sequence(
            str(self.get_parameter("sequence_json").value)
        )

        self.request_pub = self.create_publisher(
            String,
            self.action_request_topic,
            10,
        )
        self.ack_pub = self.create_publisher(
            String,
            self.action_ack_topic,
            10,
        )

        self.robot_command_sub = self.create_subscription(
            String,
            self.robot_command_topic,
            self.robot_command_callback,
            10,
        )
        self.action_ack_sub = self.create_subscription(
            String,
            self.action_ack_topic,
            self.action_ack_callback,
            10,
        )

        self.index = 0
        # Forwarded command id currently in flight (set when we see /robot/command).
        self.current_id: Optional[int] = None

        self.get_logger().info("Decision test driver started.")
        self.get_logger().info(f"Action request topic: {self.action_request_topic}")
        self.get_logger().info(f"Robot command topic:  {self.robot_command_topic}")
        self.get_logger().info(f"Action ACK topic:     {self.action_ack_topic}")
        self.get_logger().info(f"simulate_robot_ack={self.simulate_robot_ack}")
        self.get_logger().info(
            f"{len(self.sequence)} command(s) queued; first send in "
            f"{self.startup_delay_sec:.1f}s"
        )

        self._schedule_once(self.startup_delay_sec, self.send_next_command)

    def _load_sequence(self, raw: str) -> List[Dict[str, Any]]:
        text = raw.strip()
        if not text or text == "[]":
            return list(DEFAULT_SEQUENCE)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f"sequence_json is not valid JSON ({exc}); using default sequence."
            )
            return list(DEFAULT_SEQUENCE)
        if not isinstance(parsed, list):
            self.get_logger().error(
                "sequence_json must be a JSON array; using default sequence."
            )
            return list(DEFAULT_SEQUENCE)
        return parsed

    def _schedule_once(self, delay: float, fn):
        """Run fn once after `delay` seconds using a self-cancelling timer."""
        timer = None

        def _cb():
            timer.cancel()
            self.destroy_timer(timer)
            fn()

        timer = self.create_timer(delay, _cb)

    def send_next_command(self):
        if self.index >= len(self.sequence):
            if self.loop_sequence and self.sequence:
                self.get_logger().info("Sequence complete; looping.")
                self.index = 0
            else:
                self.get_logger().info("Sequence complete. Idle.")
                return

        command = self.sequence[self.index]
        human_index = self.index + 1

        msg = String()
        msg.data = json.dumps(command)
        self.request_pub.publish(msg)

        action_id = command.get("action_id")
        n_points = len(command.get("points", []) or [])
        self.get_logger().info(
            f"[{human_index}/{len(self.sequence)}] -> /decision/action_request "
            f"action_id={action_id} points={n_points}"
        )

        self.index += 1

    def robot_command_callback(self, msg: String):
        """Decision node approved a command and forwarded it to the robot."""
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid /robot/command JSON received.")
            return

        command_id = self._safe_int(command.get("id"), default=-1)
        action_id = self._safe_int(command.get("action_id"), default=-1)
        self.current_id = command_id

        self.get_logger().info(
            f"Decision approved command id={command_id} "
            f"action_id={action_id} (forwarded to robot)"
        )

        if self.simulate_robot_ack:
            self._schedule_once(
                self.sim_ack_delay_sec,
                lambda: self._send_ack(command_id, action_id),
            )

    def _send_ack(self, command_id: int, action_id: int):
        ack: Dict[str, Any] = {
            "command_id": command_id,
            "action_id": action_id,
            "status": "DONE",
        }
        # COLLECT_BLOCKS completion implies we are now holding blocks.
        if action_id == 2:
            ack["holding_blocks"] = True

        msg = String()
        msg.data = json.dumps(ack)
        self.ack_pub.publish(msg)
        self.get_logger().info(
            f"(sim robot) -> /robot/action_ack command_id={command_id} status=DONE"
        )

    def action_ack_callback(self, msg: String):
        """A DONE ACK for the in-flight command advances the sequence."""
        try:
            ack = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        command_id = self._safe_int(ack.get("command_id"), default=-1)
        status = str(ack.get("status", "")).upper()

        if command_id != self.current_id:
            return

        if status == "DONE":
            self.get_logger().info(
                f"ACK DONE for command id={command_id}; advancing in "
                f"{self.inter_command_delay_sec:.1f}s"
            )
            self.current_id = None
            self._schedule_once(
                self.inter_command_delay_sec, self.send_next_command
            )
        elif status in ("REJECTED", "FAILED", "CANCELLED", "TIMEOUT"):
            self.get_logger().warning(
                f"ACK {status} for command id={command_id}; stopping sequence."
            )
            self.current_id = None

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default


def main(args=None):
    rclpy.init(args=args)

    node = DecisionTestDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Decision test driver stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
