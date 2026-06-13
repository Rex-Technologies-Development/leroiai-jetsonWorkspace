"""Unified ROS 2 telemetry logger.

Subscribes to every interesting topic on the robot stack (policy decisions,
decision-node actions, the movement commands sent to the V5, action ACKs,
perception detections / mapped blocks, odom, serial status) and appends each
message to a single chronological JSONL file under ``logs/``.

This exists because ``ros2 topic echo`` is not usable on the robot anymore
(headless systemd run, RMW that does not deliver to a separate shell, etc).
The log file is the on-disk replacement: one JSON object per line, greppable
by topic, so you can replay exactly what happened on the ROS 2 side after a run.

All monitored topics are ``std_msgs/String`` carrying JSON, so we subscribe
generically and parse the payload back into the line when possible.
"""

import json
import os
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Default set of topics to record. Each is std_msgs/String with a JSON payload.
# Override via the ``topics`` parameter (list of strings) from the launch file.
DEFAULT_TOPICS: List[str] = [
    # --- Policy: what the brain decided ---
    "/policy/decision",
    "/policy/pending_action",
    "/policy/status",
    # --- Decision node: validated actions + outcomes ---
    "/decision/action_request",
    "/decision/status",
    "/decision/rejected",
    "/decision/command_log",
    # --- Movement: what actually went to the V5, and its ACK ---
    "/robot/command",
    "/robot/action_ack",
    # --- Perception: what blocks the robot sees ---
    "/perception/oak_front/detections",
    "/perception/oak_front/mapped_objects",
    "/perception/route_suggestion",
    # --- Pose + serial link health ---
    "/v5/odom",
    "/serial_messenger/status",
]

# Default per-topic throttle (minimum seconds between logged messages). High-rate
# topics would otherwise flood the file; everything else logs every message.
DEFAULT_THROTTLE: Dict[str, float] = {
    "/perception/oak_front/detections": 0.5,
    "/perception/oak_front/mapped_objects": 0.5,
    "/perception/route_suggestion": 0.5,
    "/v5/odom": 0.2,
    "/serial_messenger/status": 1.0,
}


class TelemetryNode(Node):
    def __init__(self):
        super().__init__("telemetry_node")

        self.declare_parameter(
            "log_dir", "~/leroiai-jetsonWorkspace/logs"
        )
        self.declare_parameter("log_file", "ros2_telemetry.jsonl")
        self.declare_parameter("topics", DEFAULT_TOPICS)
        # Throttle as a list of "topic:min_interval_sec" strings so it can be set
        # from a launch file. Left at the default -> use DEFAULT_THROTTLE. The
        # default is a single blank string (not []) because rclpy cannot infer
        # the type of an empty list parameter.
        self.declare_parameter("throttle", [""])
        # Drop in a fresh file on every node start (vs. append across runs).
        self.declare_parameter("truncate_on_start", False)

        log_dir = os.path.expanduser(str(self.get_parameter("log_dir").value))
        log_file = str(self.get_parameter("log_file").value)
        self.log_path = os.path.join(log_dir, log_file)

        self.topics = [
            str(t) for t in self.get_parameter("topics").value if str(t).strip()
        ]
        self.throttle = self._parse_throttle()
        truncate = bool(self.get_parameter("truncate_on_start").value)

        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            self.get_logger().error(
                f"Cannot create log dir {log_dir}: {exc}. Telemetry disabled."
            )
            self.log_path = ""

        if self.log_path and truncate:
            try:
                open(self.log_path, "w", encoding="utf-8").close()
            except OSError as exc:
                self.get_logger().warning(f"Could not truncate log: {exc}")

        # Last logged wall-clock time per topic, for throttling.
        self._last_logged: Dict[str, float] = {}
        self._msg_count = 0

        self._subs = []
        for topic in self.topics:
            # Closure capture: bind topic per-subscription.
            sub = self.create_subscription(
                String,
                topic,
                self._make_callback(topic),
                10,
            )
            self._subs.append(sub)

        # Run-separator banner so each launch is clearly delimited in the file
        # (we append across runs and keep the history in git).
        now = time.time()
        self._write_event(
            {
                "event": "RUN_START",
                "run_started_iso": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(now)
                ),
                "topics": self.topics,
            }
        )

        self.get_logger().info(
            f"Telemetry logger started. Logging {len(self.topics)} topics to "
            f"{self.log_path or '(disabled)'}"
        )
        for topic in self.topics:
            hz = self.throttle.get(topic)
            note = f" (throttle {hz:.2f}s)" if hz else ""
            self.get_logger().info(f"  - {topic}{note}")

    def _parse_throttle(self) -> Dict[str, float]:
        raw = self.get_parameter("throttle").value or []

        parsed: Dict[str, float] = {}
        for item in raw:
            text = str(item).strip()
            if not text or ":" not in text:
                continue
            topic, _, interval = text.rpartition(":")
            try:
                parsed[topic.strip()] = float(interval.strip())
            except ValueError:
                self.get_logger().warning(f"Ignoring bad throttle entry: {text}")

        # No usable overrides given (default blank value) -> fall back to the
        # built-in per-topic throttles.
        if not parsed:
            return dict(DEFAULT_THROTTLE)
        return parsed

    def _make_callback(self, topic: str):
        def callback(msg: String):
            self._on_message(topic, msg)

        return callback

    def _on_message(self, topic: str, msg: String):
        now = time.time()

        min_interval = self.throttle.get(topic, 0.0)
        if min_interval > 0.0:
            last = self._last_logged.get(topic, 0.0)
            if now - last < min_interval:
                return
        self._last_logged[topic] = now

        # Monitored topics carry JSON; fall back to the raw string otherwise.
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            data = msg.data

        self._write_event(
            {
                "topic": topic,
                "data": data,
            }
        )

    def _write_event(self, event: dict) -> None:
        if not self.log_path:
            return

        event.setdefault("recv_stamp_ns", self.get_clock().now().nanoseconds)
        event.setdefault("recv_time_unix", time.time())

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
            self._msg_count += 1
        except OSError as exc:
            # Never let logging crash the node.
            self.get_logger().warning(f"Failed to write telemetry: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Telemetry logger stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
