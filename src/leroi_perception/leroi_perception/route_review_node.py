import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Verdicts the operator can record for a settled decision.
VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"
VERDICT_SKIPPED = "skipped"


class RouteReviewNode(Node):
    """
    Interactive reviewer for the route planner.

    Loop (one decision at a time):
      1. OBSERVE   -- watch the camera-fed route suggestions for `observe_sec`
                      seconds and settle on the most recent valid decision.
      2. REPORT    -- print the running game state (own half) + the settled
                      route decision to the terminal.
      3. CONFIRM   -- ask the operator "correct? [y/n]" and log the verdict.
      4. ADVANCE   -- wait for the operator to move the robot, then press Enter
                      to evaluate the next scene. Loop back to OBSERVE.

    It does NOT drive the robot. It only reads:
      /perception/route_suggestion   -- the planner's COLLECT route
      /perception/oak_front/detections -- raw field-projected blocks (game state)

    Operator input is read from the controlling terminal (/dev/tty) so it keeps
    working under `ros2 launch`, where stdin may be redirected. A confirm topic
    (default /route_review/confirm) is also accepted as a fallback for headless
    runs:  ros2 topic pub --once /route_review/confirm std_msgs/String "{data: 'y'}"
    """

    def __init__(self):
        super().__init__("route_review_node")

        # Topics
        self.declare_parameter(
            "route_suggestion_topic", "/perception/route_suggestion"
        )
        self.declare_parameter(
            "detections_topic", "/perception/oak_front/detections"
        )
        self.declare_parameter("confirm_topic", "/route_review/confirm")

        # Behavior
        self.declare_parameter("observe_sec", 3.0)
        self.declare_parameter("alliance_color", "blue")
        # Which half of the field this robot owns: "left" (x < center) or
        # "right" (x >= center). 24in plays left, 15in plays right.
        self.declare_parameter("own_half", "left")
        self.declare_parameter("field_center_x_in", 72.0)
        self.declare_parameter("field_max_x_in", 144.0)
        self.declare_parameter("field_max_y_in", 144.0)
        # Where to append the review log (one JSON object per reviewed decision).
        self.declare_parameter(
            "log_path",
            os.path.expanduser(
                "~/leroiai-jetsonWorkspace/logs/route_review.jsonl"
            ),
        )

        self.route_suggestion_topic = str(
            self.get_parameter("route_suggestion_topic").value
        )
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.confirm_topic = str(self.get_parameter("confirm_topic").value)

        self.observe_sec = float(self.get_parameter("observe_sec").value)
        self.alliance_color = (
            str(self.get_parameter("alliance_color").value).lower().strip()
        )
        self.opponent_color = "red" if self.alliance_color == "blue" else "blue"
        self.own_half = str(self.get_parameter("own_half").value).lower().strip()
        self.field_center_x_in = float(
            self.get_parameter("field_center_x_in").value
        )
        self.field_max_x_in = float(self.get_parameter("field_max_x_in").value)
        self.field_max_y_in = float(self.get_parameter("field_max_y_in").value)
        self.log_path = str(self.get_parameter("log_path").value)

        # Shared state updated by ROS callbacks, read by the review thread.
        self._lock = threading.Lock()
        self._latest_route: Optional[Dict[str, Any]] = None
        self._latest_route_time = 0.0
        self._latest_detections: Optional[Dict[str, Any]] = None
        self._latest_detections_time = 0.0

        # Confirm answers arriving over the topic fallback.
        self._pending_answer: Optional[str] = None

        # Running game-state snapshot, updated each reviewed cycle.
        self.cycle_index = 0
        self.game_state: Dict[str, Any] = {
            "cycle": 0,
            "robot": None,
            "own_blocks": 0,
            "opp_blocks": 0,
            "last_decision": None,
        }

        self.create_subscription(
            String, self.route_suggestion_topic, self._route_cb, 10
        )
        self.create_subscription(
            String, self.detections_topic, self._detections_cb, 10
        )
        self.create_subscription(
            String, self.confirm_topic, self._confirm_cb, 10
        )

        self._tty = self._open_tty()

        self.get_logger().info("Route review node started.")
        self.get_logger().info(
            f"alliance={self.alliance_color} own_half={self.own_half} "
            f"observe_sec={self.observe_sec}"
        )

        # Drive the interactive loop off the ROS thread.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._review_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ ROS

    def _route_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            self._latest_route = data
            self._latest_route_time = time.time()

    def _detections_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            self._latest_detections = data
            self._latest_detections_time = time.time()

    def _confirm_cb(self, msg: String):
        with self._lock:
            self._pending_answer = str(msg.data).strip().lower()

    # ----------------------------------------------------------------- TTY

    def _open_tty(self):
        """Open the controlling terminal for blocking operator input.

        Returns a readable file object, or None if no terminal is available
        (headless run) -- in that case input comes only from the confirm topic.
        """
        try:
            return open("/dev/tty", "r")
        except OSError:
            try:
                import sys

                if sys.stdin and sys.stdin.isatty():
                    return sys.stdin
            except Exception:
                pass
        return None

    def _say(self, text: str = ""):
        # Print to screen (captured by launch) and flush so prompts appear
        # immediately even when interleaved with other nodes' output.
        print(text, flush=True)

    def _ask(self, prompt: str, valid: Optional[List[str]] = None) -> str:
        """Block until the operator answers via the terminal or confirm topic.

        Returns the lowercased answer. Empty string means "just pressed Enter".
        """
        self._say(prompt)
        while not self._stop.is_set():
            # Topic fallback first (non-blocking).
            with self._lock:
                answer = self._pending_answer
                self._pending_answer = None
            if answer is not None:
                if valid is None or answer in valid or answer == "":
                    return answer
                self._say(f"  (ignored '{answer}', expected one of {valid})")
                self._say(prompt)
                continue

            # Terminal read. readline blocks; guard with the tty being present.
            if self._tty is not None:
                line = self._tty.readline()
                if line == "":
                    # EOF on the tty -- fall back to topic-only.
                    self._tty = None
                    continue
                answer = line.strip().lower()
                if valid is None or answer in valid or answer == "":
                    return answer
                self._say(f"  (expected one of {valid})")
                self._say(prompt)
            else:
                # No terminal: poll the topic.
                time.sleep(0.1)
        return ""

    # -------------------------------------------------------------- Review

    def _snapshot(self):
        with self._lock:
            route = self._latest_route
            route_age = (
                None
                if self._latest_route_time <= 0.0
                else time.time() - self._latest_route_time
            )
            detections = self._latest_detections
            det_age = (
                None
                if self._latest_detections_time <= 0.0
                else time.time() - self._latest_detections_time
            )
        return route, route_age, detections, det_age

    def _on_own_half(self, x: float) -> bool:
        if self.own_half == "right":
            return x >= self.field_center_x_in
        return x < self.field_center_x_in

    def _count_blocks(self, detections: Optional[Dict[str, Any]]):
        """Count own/opponent blocks the camera sees on this robot's half."""
        own = 0
        opp = 0
        blocks: List[Dict[str, Any]] = []
        if not detections:
            return own, opp, blocks
        for det in detections.get("detections", []):
            if not isinstance(det, dict):
                continue
            name = str(det.get("class_name", det.get("class", ""))).lower()
            if "red" in name:
                color = "red"
            elif "blue" in name:
                color = "blue"
            else:
                continue
            x = det.get("x_field_in")
            y = det.get("y_field_in")
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not self._on_own_half(x):
                continue
            if color == self.alliance_color:
                own += 1
            elif color == self.opponent_color:
                opp += 1
            blocks.append({"color": color, "x": round(x, 1), "y": round(y, 1)})
        return own, opp, blocks

    def _observe(self) -> Dict[str, Any]:
        """Watch route suggestions for observe_sec and return the settled one."""
        self._say(
            f"\nObserving the field for {self.observe_sec:.0f}s "
            f"(cycle {self.cycle_index + 1})..."
        )
        deadline = time.time() + self.observe_sec
        settled: Optional[Dict[str, Any]] = None
        while time.time() < deadline and not self._stop.is_set():
            route, route_age, _, _ = self._snapshot()
            # Keep the freshest route that actually planned something, else the
            # freshest packet at all (so we can report NO_ROUTE / NO_VALID_ODOM).
            if route is not None and route_age is not None and route_age < 1.0:
                if route.get("reason") == "OK" or settled is None:
                    settled = route
            time.sleep(0.1)
        route, route_age, detections, det_age = self._snapshot()
        if route is not None and (settled is None or route.get("reason") == "OK"):
            settled = route
        return {
            "route": settled,
            "route_age": route_age,
            "detections": detections,
            "det_age": det_age,
        }

    def _update_game_state(self, snap: Dict[str, Any]):
        route = snap.get("route") or {}
        own, opp, blocks = self._count_blocks(snap.get("detections"))
        robot = None
        if route.get("robot_pose_valid"):
            robot = {
                "x": round(float(route.get("robot_x", 0.0)), 1),
                "y": round(float(route.get("robot_y", 0.0)), 1),
                "heading": round(float(route.get("robot_heading_deg", 0.0)), 1),
            }
        self.cycle_index += 1
        self.game_state = {
            "cycle": self.cycle_index,
            "robot": robot,
            "own_blocks": own,
            "opp_blocks": opp,
            "own_half_blocks": blocks,
            "last_decision": {
                "reason": route.get("reason"),
                "points": route.get("points", []),
                "candidate_count": route.get("candidate_count"),
            },
        }
        return own, opp, blocks

    def _report(self, snap: Dict[str, Any], own: int, opp: int):
        route = snap.get("route")
        bar = "=" * 64
        self._say("\n" + bar)
        self._say(f"DECISION  (cycle {self.cycle_index})")
        self._say(bar)

        if route is None:
            self._say("No route suggestion received yet. Is perception running?")
            return

        reason = route.get("reason", "?")
        gs = self.game_state
        robot = gs.get("robot")
        if robot:
            self._say(
                f"Robot pose : x={robot['x']} y={robot['y']} "
                f"heading={robot['heading']} deg"
            )
        else:
            self._say("Robot pose : (none / not valid)")
        self._say(
            f"Alliance   : {self.alliance_color}  |  own half: {self.own_half} "
            f"(x {'>=' if self.own_half == 'right' else '<'} "
            f"{self.field_center_x_in:.0f})"
        )
        self._say(
            f"Game state : {own} {self.alliance_color} block(s), "
            f"{opp} {self.opponent_color} block(s) on our half  "
            f"(candidates seen: {route.get('candidate_count')})"
        )
        self._say(f"Reason     : {reason}")

        points = route.get("points", [])
        if points:
            self._say(f"Route      : {len(points)} waypoint(s) (COLLECT_BLOCKS)")
            debug = {
                tuple(round(d.get(k, 0.0), 2) for k in ("x", "y")): d
                for d in route.get("route_debug", [])
            }
            for i, p in enumerate(points, 1):
                key = (round(p.get("x", 0.0), 2), round(p.get("y", 0.0), 2))
                extra = debug.get(key, {})
                score = extra.get("score")
                count = extra.get("source_count")
                tail = ""
                if score is not None:
                    tail = f"  score={score} blocks={count}"
                self._say(
                    f"   {i}. x={p.get('x'):.1f}  y={p.get('y'):.1f}  "
                    f"heading={p.get('heading'):.1f} deg{tail}"
                )
        else:
            self._say("Route      : (empty -- no pickup target)")
        self._say(bar)

    def _log_result(self, snap: Dict[str, Any], verdict: str, note: str):
        route = snap.get("route") or {}
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cycle": self.cycle_index,
            "verdict": verdict,
            "note": note,
            "alliance_color": self.alliance_color,
            "own_half": self.own_half,
            "game_state": self.game_state,
            "reason": route.get("reason"),
            "points": route.get("points", []),
            "candidate_count": route.get("candidate_count"),
        }
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            self.get_logger().warning(f"Could not write review log: {exc}")

    def _review_loop(self):
        self._say("\nRoute planner review. Move the robot between scenes.")
        self._say(
            "Confirm prompts accept y/n; you can also publish to "
            f"{self.confirm_topic}.\n"
        )
        while not self._stop.is_set():
            snap = self._observe()
            own, opp, _ = self._update_game_state(snap)
            self._report(snap, own, opp)

            answer = self._ask(
                "Is this decision correct? [y]es / [n]o / [r]e-observe / [q]uit: ",
                valid=["y", "yes", "n", "no", "r", "re", "q", "quit", ""],
            )
            if answer in ("q", "quit"):
                self._say("Quitting review.")
                self._shutdown()
                return
            if answer in ("r", "re", ""):
                # Re-observe the same scene without logging a verdict.
                continue

            verdict = (
                VERDICT_CORRECT if answer in ("y", "yes") else VERDICT_INCORRECT
            )
            note = ""
            if verdict == VERDICT_INCORRECT:
                note = self._ask("  What was wrong? (optional, Enter to skip): ")
            self._log_result(snap, verdict, note)
            self._say(f"Recorded: {verdict}. Logged to {self.log_path}")

            nxt = self._ask(
                "\nMove the robot, then press Enter for the next scene "
                "([q] to quit): ",
                valid=["", "q", "quit"],
            )
            if nxt in ("q", "quit"):
                self._say("Quitting review.")
                self._shutdown()
                return

    def _shutdown(self):
        self._stop.set()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RouteReviewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Route review stopped by user.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
