"""Policy node: drives the robot via a pluggable strategy.

Two interchangeable strategies (select with the ``strategy_mode`` param):

  - rl       : run an exported RL model (ONNX) and pick the argmax action.
  - scripted : execute a fixed, repeatable game plan (collect -> score -> scan
               -> descore-if-opponent -> loop). No learning.

Both feed the same pipeline:

  game state -> strategy.propose() -> proposed action
            -> (operator confirm/cancel gate) -> /decision/action_request

During testing, require_confirmation=True holds each proposed action on
/policy/pending_action until an operator publishes a decision on /policy/decision
('confirm' or 'cancel'). The confirm may carry points for point-based actions:

    ros2 topic pub --once /policy/decision std_msgs/String "data: 'confirm 250,200,90 270,220,45'"
    ros2 topic pub --once /policy/decision std_msgs/String "data: 'cancel'"
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from leroi_policy.strategies import (
    POINT_ACTIONS,
    ActionProposal,
    GameState,
    action_name,
    extract_blocks,
    make_strategy,
)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")

        # --- Topics ---
        self.declare_parameter("mapped_objects_topic", "/perception/oak_front/mapped_objects")
        self.declare_parameter("detections_topic", "/perception/oak_front/detections")
        self.declare_parameter("route_suggestion_topic", "/perception/route_suggestion")
        self.declare_parameter("odom_topic", "/v5/odom")
        self.declare_parameter("action_request_topic", "/decision/action_request")
        self.declare_parameter("pending_action_topic", "/policy/pending_action")
        self.declare_parameter("decision_topic", "/policy/decision")
        self.declare_parameter("status_topic", "/policy/status")
        self.declare_parameter("decision_status_topic", "/decision/status")

        # --- Strategy selection ---
        # "rl" -> RL model (ONNX). "scripted" -> defined game plan.
        self.declare_parameter("strategy_mode", "rl")

        # --- RL strategy params ---
        self.declare_parameter("model_path", "~/leroiai-jetsonWorkspace/models/policy/policy.onnx")
        self.declare_parameter("onnx_providers", "CPUExecutionProvider")
        self.declare_parameter("intra_op_num_threads", 1)
        self.declare_parameter("obs_dim_override", 0)
        self.declare_parameter("action_id_map", "")

        # --- Scripted strategy params ---
        self.declare_parameter("alliance_color", "blue")
        # Action ids; 0 -> alliance-aware default.
        self.declare_parameter("score_long_action_id", 0)
        self.declare_parameter("score_midlow_action_id", 0)
        self.declare_parameter("descore_action_id", 0)
        self.declare_parameter("max_collect_targets", 3)
        # Alliance-specific positions (robot units = inches), JSON arrays.
        self.declare_parameter("scan_point_blue_json", "")
        self.declare_parameter("scan_point_red_json", "")
        self.declare_parameter("collect_fallback_blue_json", "")
        self.declare_parameter("collect_fallback_red_json", "")
        # Goal-state assessment.
        self.declare_parameter("goal_regions_json", "")
        self.declare_parameter("goal_color_margin", 1)
        self.declare_parameter("goal_empty_max", 1)
        self.declare_parameter("goal_full_min", 4)
        # Collect routing + zone.
        self.declare_parameter("use_route_planner_for_collect", True)
        self.declare_parameter("zone_x_max_in", 0.0)      # 0 -> no zone limit
        self.declare_parameter("zone_x_wander_in", 12.0)
        # Perception is cm, V5/odom is inches. 1/2.54 converts collect targets.
        self.declare_parameter("perception_to_robot_scale", 1.0 / 2.54)

        # --- Behavior ---
        self.declare_parameter("inference_rate_hz", 1.0)
        self.declare_parameter("require_confirmation", True)
        self.declare_parameter("respect_decision_busy", True)
        self.declare_parameter("max_blocks", 10)
        self.declare_parameter("field_size_cm", 365.76)

        gp = self.get_parameter
        self.mapped_objects_topic = str(gp("mapped_objects_topic").value)
        self.detections_topic = str(gp("detections_topic").value)
        self.route_suggestion_topic = str(gp("route_suggestion_topic").value)
        self.odom_topic = str(gp("odom_topic").value)
        self.action_request_topic = str(gp("action_request_topic").value)
        self.pending_action_topic = str(gp("pending_action_topic").value)
        self.decision_topic = str(gp("decision_topic").value)
        self.status_topic = str(gp("status_topic").value)
        self.decision_status_topic = str(gp("decision_status_topic").value)

        self.strategy_mode = str(gp("strategy_mode").value)
        self.inference_rate_hz = float(gp("inference_rate_hz").value)
        self.require_confirmation = bool(gp("require_confirmation").value)
        self.respect_decision_busy = bool(gp("respect_decision_busy").value)
        self.max_blocks = int(gp("max_blocks").value)
        self.field_size_cm = float(gp("field_size_cm").value)

        # Strategy config: a flat dict each strategy reads what it needs from.
        # Action ids pass through as-is; 0 lets the scripted strategy pick the
        # alliance-aware default.
        strategy_cfg: Dict[str, Any] = {
            # RL
            "model_path": str(gp("model_path").value),
            "onnx_providers": str(gp("onnx_providers").value),
            "intra_op_num_threads": int(gp("intra_op_num_threads").value),
            "obs_dim_override": int(gp("obs_dim_override").value),
            "action_id_map": str(gp("action_id_map").value),
            "max_blocks": self.max_blocks,
            "field_size_cm": self.field_size_cm,
            # Scripted
            "alliance_color": str(gp("alliance_color").value),
            "score_long_action_id": int(gp("score_long_action_id").value),
            "score_midlow_action_id": int(gp("score_midlow_action_id").value),
            "descore_action_id": int(gp("descore_action_id").value),
            "max_collect_targets": int(gp("max_collect_targets").value),
            "scan_point_blue_json": str(gp("scan_point_blue_json").value),
            "scan_point_red_json": str(gp("scan_point_red_json").value),
            "collect_fallback_blue_json": str(gp("collect_fallback_blue_json").value),
            "collect_fallback_red_json": str(gp("collect_fallback_red_json").value),
            "goal_regions_json": str(gp("goal_regions_json").value),
            "goal_color_margin": int(gp("goal_color_margin").value),
            "goal_empty_max": int(gp("goal_empty_max").value),
            "goal_full_min": int(gp("goal_full_min").value),
            "use_route_planner_for_collect": bool(gp("use_route_planner_for_collect").value),
            "zone_x_max_in": float(gp("zone_x_max_in").value),
            "zone_x_wander_in": float(gp("zone_x_wander_in").value),
            "perception_to_robot_scale": float(gp("perception_to_robot_scale").value),
        }

        # --- State (latest cached perception/pose) ---
        self.latest_mapped: Optional[Dict[str, Any]] = None
        self.latest_detections: Optional[Dict[str, Any]] = None
        self.latest_route: Optional[Dict[str, Any]] = None
        self.latest_odom: Optional[Dict[str, Any]] = None
        self.decision_waiting_for_ack = False

        self.waiting_for_decision = False
        self.pending_action: Optional[ActionProposal] = None
        self.next_command_id = 1

        # --- Publishers ---
        self.action_request_pub = self.create_publisher(String, self.action_request_topic, 10)
        self.pending_action_pub = self.create_publisher(String, self.pending_action_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        # --- Subscriptions ---
        self.create_subscription(String, self.mapped_objects_topic, self.mapped_objects_cb, 10)
        self.create_subscription(String, self.detections_topic, self.detections_cb, 10)
        self.create_subscription(String, self.route_suggestion_topic, self.route_suggestion_cb, 10)
        self.create_subscription(String, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(String, self.decision_topic, self.decision_cb, 10)
        self.create_subscription(String, self.decision_status_topic, self.decision_status_cb, 10)

        # --- Build strategy ---
        try:
            self.strategy = make_strategy(self.strategy_mode, strategy_cfg, self.get_logger())
        except ValueError as exc:
            self.get_logger().error(f"{exc} Falling back to scripted strategy.")
            self.strategy = make_strategy("scripted", strategy_cfg, self.get_logger())

        # --- Timers ---
        period = 1.0 / self.inference_rate_hz if self.inference_rate_hz > 0 else 1.0
        self.tick_timer = self.create_timer(period, self.strategy_tick)
        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info("Policy node started.")
        self.get_logger().info(f"strategy_mode: {self.strategy_mode} (-> {self.strategy.name})")
        self.get_logger().info(f"require_confirmation: {self.require_confirmation}")
        self.get_logger().info(
            f"Confirm/cancel on '{self.decision_topic}' "
            f"(e.g. ros2 topic pub --once {self.decision_topic} std_msgs/String \"data: 'confirm'\")"
        )

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #
    def mapped_objects_cb(self, msg: String):
        self.latest_mapped = self._safe_json(msg.data)

    def detections_cb(self, msg: String):
        self.latest_detections = self._safe_json(msg.data)

    def route_suggestion_cb(self, msg: String):
        data = self._safe_json(msg.data)
        if data is not None:
            self.latest_route = data

    def odom_cb(self, msg: String):
        data = self._safe_json(msg.data)
        if data is not None:
            self.latest_odom = data

    def decision_status_cb(self, msg: String):
        data = self._safe_json(msg.data)
        if data is not None:
            self.decision_waiting_for_ack = bool(data.get("waiting_for_ack", False))

    def decision_cb(self, msg: String):
        """Operator confirm/cancel for the currently pending action."""
        if not self.waiting_for_decision or self.pending_action is None:
            self.get_logger().warning("Got a decision but no action is pending; ignoring.")
            return

        decision, points = self._parse_decision(msg.data)

        if decision == "confirm":
            self._send_action(self.pending_action, points)
            self.waiting_for_decision = False
            self.pending_action = None
        elif decision == "cancel":
            self.get_logger().info(
                f"Operator CANCELLED action {self.pending_action.action_id} "
                f"({self.pending_action.action_name})."
            )
            self.waiting_for_decision = False
            self.pending_action = None
        else:
            self.get_logger().warning(
                f"Unrecognized decision '{msg.data}'. Use 'confirm' or 'cancel' "
                f"(optionally with points, e.g. 'confirm 250,200,90')."
            )

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def strategy_tick(self):
        if self.waiting_for_decision:
            return
        if self.respect_decision_busy and self.decision_waiting_for_ack:
            return

        state = self._build_state()
        proposal = self.strategy.propose(state)
        if proposal is None:
            return

        self.pending_action = proposal
        self.waiting_for_decision = self.require_confirmation
        self._publish_pending(proposal)

        if self.require_confirmation:
            needs_pts = proposal.action_id in POINT_ACTIONS and not proposal.points
            self.get_logger().info(
                f"PROPOSED [{proposal.source or self.strategy.name}] "
                f"action_id={proposal.action_id} ({proposal.action_name}) "
                f"points={len(proposal.points)}"
                + (
                    "  -> needs points; confirm e.g. 'confirm 250,200,90'"
                    if needs_pts
                    else "  -> publish 'confirm' or 'cancel'"
                )
            )
        else:
            # Live mode: auto-run with the strategy's own points.
            self._send_action(proposal, proposal.points)
            self.pending_action = None

    def _build_state(self) -> GameState:
        return GameState(
            odom=self.latest_odom,
            mapped_objects=self.latest_mapped,
            detections=self.latest_detections,
            route_suggestion=self.latest_route,
            blocks=extract_blocks(self.latest_mapped),
            decision_busy=self.decision_waiting_for_ack,
            now_ns=self.get_clock().now().nanoseconds,
        )

    # ------------------------------------------------------------------ #
    # Action output
    # ------------------------------------------------------------------ #
    def _send_action(self, proposal: ActionProposal, points: List[Dict[str, float]]):
        # Operator-supplied points (at confirm) override the strategy's; otherwise
        # fall back to whatever the strategy proposed.
        effective_points = points if points else list(proposal.points)
        action_id = int(proposal.action_id)

        if action_id in POINT_ACTIONS and not effective_points:
            self.get_logger().warning(
                f"Action {action_id} ({proposal.action_name}) usually needs points "
                f"but none were given; sending with empty points."
            )

        command = {
            "type": "ACTION",
            "id": self._allocate_command_id(),
            "action_id": action_id,
            "points": effective_points,
            "params": {
                "source": proposal.source or self.strategy.name,
                "confidence": proposal.confidence,
                **proposal.params,
            },
        }

        msg = String()
        msg.data = json.dumps(command)
        self.action_request_pub.publish(msg)

        self.get_logger().info(
            f"SENT to decision: action_id={action_id} ({proposal.action_name}) "
            f"points={len(effective_points)} source={proposal.source or self.strategy.name}"
        )

    def _allocate_command_id(self) -> int:
        cid = self.next_command_id
        self.next_command_id += 1
        return cid

    def _publish_pending(self, proposal: ActionProposal):
        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "action_id": proposal.action_id,
            "action_name": proposal.action_name,
            "points": proposal.points,
            "confidence": proposal.confidence,
            "source": proposal.source or self.strategy.name,
            "needs_points": proposal.action_id in POINT_ACTIONS and not proposal.points,
            "info": proposal.info,
            "instructions": (
                f"Publish to {self.decision_topic}: 'confirm' to run, 'cancel' to skip. "
                f"For point actions, append points: 'confirm x,y,heading x,y,heading'."
            ),
        }
        msg = String()
        msg.data = json.dumps(packet)
        self.pending_action_pub.publish(msg)

    def publish_status(self):
        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "strategy_mode": self.strategy_mode,
            "strategy": self.strategy.name,
            "waiting_for_decision": self.waiting_for_decision,
            "decision_waiting_for_ack": self.decision_waiting_for_ack,
            "pending_action": (
                {
                    "action_id": self.pending_action.action_id,
                    "action_name": self.pending_action.action_name,
                }
                if self.pending_action
                else None
            ),
            "strategy_status": self.strategy.status(),
        }
        msg = String()
        msg.data = json.dumps(packet)
        self.status_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    def _parse_decision(self, raw: str) -> Tuple[Optional[str], List[Dict[str, float]]]:
        text = (raw or "").strip()
        if not text:
            return None, []

        if text.startswith("{"):
            obj = self._safe_json(text) or {}
            word = str(obj.get("decision", obj.get("action", ""))).lower()
            return self._decision_word(word), self._normalize_points(obj.get("points", []))

        tokens = text.split()
        word = self._decision_word(tokens[0].lower())
        points = self._parse_point_tokens(tokens[1:]) if len(tokens) > 1 else []
        return word, points

    @staticmethod
    def _decision_word(word: str) -> Optional[str]:
        if word in ("confirm", "yes", "y", "ok", "run", "go", "true", "1", "c"):
            return "confirm"
        if word in ("cancel", "no", "n", "deny", "skip", "reject", "false", "0", "x"):
            return "cancel"
        return None

    def _parse_point_tokens(self, tokens: List[str]) -> List[Dict[str, float]]:
        joined = " ".join(tokens).replace(";", " ")
        points = []
        for g in (g for g in joined.split() if g.strip()):
            parts = [p for p in g.split(",") if p.strip() != ""]
            try:
                x = float(parts[0])
                y = float(parts[1])
                heading = float(parts[2]) if len(parts) >= 3 else 0.0
                points.append({"x": x, "y": y, "heading": heading})
            except (IndexError, ValueError):
                self.get_logger().warning(f"Skipping unparseable point '{g}'.")
        return points

    def _normalize_points(self, raw_points: Any) -> List[Dict[str, float]]:
        out = []
        if not isinstance(raw_points, list):
            return out
        for p in raw_points:
            try:
                if isinstance(p, dict):
                    out.append(
                        {"x": float(p["x"]), "y": float(p["y"]), "heading": float(p.get("heading", 0.0))}
                    )
                elif isinstance(p, (list, tuple)):
                    out.append(
                        {"x": float(p[0]), "y": float(p[1]), "heading": float(p[2]) if len(p) >= 3 else 0.0}
                    )
            except (KeyError, IndexError, ValueError, TypeError):
                continue
        return out

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Policy node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
