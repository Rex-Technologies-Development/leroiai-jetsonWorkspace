import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import onnxruntime as ort


# Mirror of leroi_decision/decision_node.py ACTION_NAMES. Kept local so this
# package does not import leroi_decision. If you change one, change both.
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
}

# Actions that consume target points. Everything else auto-navigates on the V5.
POINT_ACTIONS = {1, 2}  # MOVE_CHASSIS_ONLY, COLLECT_BLOCKS


class PolicyNode(Node):
    """Runs an exported RL policy (ONNX) live and proposes robot actions.

    Pipeline:
      perception + odom  ->  build_observation()  ->  ONNX policy  ->  argmax
      ->  proposed action  ->  (confirm/cancel gate)  ->  /decision/action_request

    During testing, require_confirmation=True: each proposed action is published
    on /policy/pending_action and held until an operator publishes a decision on
    /policy/decision ('confirm' or 'cancel'). The confirm message may carry target
    points for point-based actions, e.g.:

        ros2 topic pub --once /policy/decision std_msgs/String "data: 'confirm 250,200,90 270,220,45'"
        ros2 topic pub --once /policy/decision std_msgs/String "data: 'cancel'"
    """

    def __init__(self):
        super().__init__("policy_node")

        # --- Topics ---
        self.declare_parameter("mapped_objects_topic", "/perception/oak_front/mapped_objects")
        self.declare_parameter("detections_topic", "/perception/oak_front/detections")
        self.declare_parameter("odom_topic", "/v5/odom")
        self.declare_parameter("action_request_topic", "/decision/action_request")
        self.declare_parameter("pending_action_topic", "/policy/pending_action")
        self.declare_parameter("decision_topic", "/policy/decision")
        self.declare_parameter("status_topic", "/policy/status")
        self.declare_parameter("decision_status_topic", "/decision/status")

        # --- Model ---
        self.declare_parameter(
            "model_path",
            "~/leroiai-jetsonWorkspace/models/policy/policy.onnx",
        )
        # ONNX execution providers, in priority order (comma-separated). Default
        # CPU: the policy is a tiny MLP at a low rate, so CPU is plenty and keeps
        # the GPU free for perception's YOLO TensorRT engine. To use the GPU you
        # must first install NVIDIA's Jetson onnxruntime-gpu build, then set e.g.
        # "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider".
        # Unavailable providers are dropped automatically (CPU is always kept).
        self.declare_parameter("onnx_providers", "CPUExecutionProvider")
        # Setting the thread count explicitly avoids onnxruntime trying to pin CPU
        # affinity, which fails on Jetson/Tegra (pthread_setaffinity_np errors).
        # 1 is plenty for a small policy MLP.
        self.declare_parameter("intra_op_num_threads", 1)
        # Override observation length if you cannot infer it from the model.
        # 0 = trust the model's declared input shape.
        self.declare_parameter("obs_dim_override", 0)
        # Map policy output index -> action_id. Empty = identity (index == action_id).
        self.declare_parameter("action_id_map", "")

        # --- Behavior ---
        self.declare_parameter("inference_rate_hz", 1.0)
        self.declare_parameter("require_confirmation", True)
        self.declare_parameter("respect_decision_busy", True)
        # Observation builder tuning (see build_observation()).
        self.declare_parameter("max_blocks", 10)
        self.declare_parameter("field_size_cm", 365.76)

        self.mapped_objects_topic = str(self.get_parameter("mapped_objects_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.action_request_topic = str(self.get_parameter("action_request_topic").value)
        self.pending_action_topic = str(self.get_parameter("pending_action_topic").value)
        self.decision_topic = str(self.get_parameter("decision_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.decision_status_topic = str(self.get_parameter("decision_status_topic").value)

        self.model_path = os.path.expanduser(str(self.get_parameter("model_path").value))
        self.onnx_providers = self._resolve_providers(
            str(self.get_parameter("onnx_providers").value)
        )
        self.intra_op_num_threads = int(self.get_parameter("intra_op_num_threads").value)
        self.obs_dim_override = int(self.get_parameter("obs_dim_override").value)
        self.action_id_map = self._parse_action_id_map(
            str(self.get_parameter("action_id_map").value)
        )

        self.inference_rate_hz = float(self.get_parameter("inference_rate_hz").value)
        self.require_confirmation = bool(self.get_parameter("require_confirmation").value)
        self.respect_decision_busy = bool(self.get_parameter("respect_decision_busy").value)
        self.max_blocks = int(self.get_parameter("max_blocks").value)
        self.field_size_cm = float(self.get_parameter("field_size_cm").value)

        # --- State (latest cached perception/pose) ---
        self.latest_mapped: Optional[Dict[str, Any]] = None
        self.latest_detections: Optional[Dict[str, Any]] = None
        self.latest_odom: Optional[Dict[str, Any]] = None
        self.decision_waiting_for_ack = False

        self.waiting_for_decision = False
        self.pending_action: Optional[Dict[str, Any]] = None
        self.obs_warned = False
        self.next_command_id = 1

        # --- Publishers ---
        self.action_request_pub = self.create_publisher(String, self.action_request_topic, 10)
        self.pending_action_pub = self.create_publisher(String, self.pending_action_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        # --- Subscriptions ---
        self.create_subscription(String, self.mapped_objects_topic, self.mapped_objects_cb, 10)
        self.create_subscription(String, self.detections_topic, self.detections_cb, 10)
        self.create_subscription(String, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(String, self.decision_topic, self.decision_cb, 10)
        self.create_subscription(String, self.decision_status_topic, self.decision_status_cb, 10)

        # --- Load model ---
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.obs_dim: int = 0
        self._load_model()

        # --- Inference timer ---
        period = 1.0 / self.inference_rate_hz if self.inference_rate_hz > 0 else 1.0
        self.infer_timer = self.create_timer(period, self.inference_tick)

        self.get_logger().info("Policy node started.")
        self.get_logger().info(f"model_path: {self.model_path}")
        self.get_logger().info(f"obs_dim: {self.obs_dim}")
        self.get_logger().info(f"require_confirmation: {self.require_confirmation}")
        self.get_logger().info(
            f"Confirm/cancel on '{self.decision_topic}' "
            f"(e.g. ros2 topic pub --once {self.decision_topic} std_msgs/String "
            f"\"data: 'confirm'\")"
        )

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.get_logger().error(
                f"Policy model not found: {self.model_path}. "
                f"Node will idle until a model is available."
            )
            return

        sess_options = ort.SessionOptions()
        if self.intra_op_num_threads > 0:
            sess_options.intra_op_num_threads = self.intra_op_num_threads

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.onnx_providers,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load ONNX model: {exc}")
            self.session = None
            return

        self.get_logger().info(
            f"ONNX running on providers: {self.session.get_providers()}"
        )

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name

        # Infer observation length from the model's declared input shape
        # (last non-batch dim). Fall back to override / default if dynamic.
        declared = None
        if inp.shape:
            last = inp.shape[-1]
            if isinstance(last, int) and last > 0:
                declared = last

        if self.obs_dim_override > 0:
            self.obs_dim = self.obs_dim_override
        elif declared is not None:
            self.obs_dim = declared
        else:
            # Dynamic input shape and no override: use the default layout length.
            self.obs_dim = self._default_obs_length()

        self.get_logger().info(
            f"Loaded policy: input='{self.input_name}' shape={inp.shape} -> "
            f"output='{self.output_name}' shape={out.shape} | obs_dim={self.obs_dim}"
        )

    def _resolve_providers(self, raw: str) -> List[str]:
        """Keep only requested providers that this onnxruntime build supports.

        Always guarantees CPUExecutionProvider as a fallback so the node runs even
        if a requested GPU provider (CUDA/TensorRT) isn't installed.
        """
        available = ort.get_available_providers()
        requested = [p.strip() for p in raw.split(",") if p.strip()]
        resolved = [p for p in requested if p in available]

        dropped = [p for p in requested if p not in available]
        if dropped:
            self.get_logger().warning(
                f"Requested ONNX providers not available and dropped: {dropped}. "
                f"Available: {available}. (For GPU on Jetson, install NVIDIA's "
                f"onnxruntime-gpu build.)"
            )

        if "CPUExecutionProvider" not in resolved:
            resolved.append("CPUExecutionProvider")
        return resolved

    def _parse_action_id_map(self, raw: str) -> Optional[List[int]]:
        raw = raw.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return [int(x) for x in parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            self.get_logger().warning(
                f"Invalid action_id_map '{raw}'; using identity mapping."
            )
            return None

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #
    def mapped_objects_cb(self, msg: String):
        self.latest_mapped = self._safe_json(msg.data)

    def detections_cb(self, msg: String):
        self.latest_detections = self._safe_json(msg.data)

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
                f"Operator CANCELLED action "
                f"{self.pending_action['action_id']} "
                f"({self.pending_action['action_name']})."
            )
            self.waiting_for_decision = False
            self.pending_action = None
        else:
            self.get_logger().warning(
                f"Unrecognized decision '{msg.data}'. Use 'confirm' or 'cancel' "
                f"(optionally with points, e.g. 'confirm 250,200,90')."
            )

    # ------------------------------------------------------------------ #
    # Inference loop
    # ------------------------------------------------------------------ #
    def inference_tick(self):
        if self.session is None:
            return
        if self.waiting_for_decision:
            return
        if self.respect_decision_busy and self.decision_waiting_for_ack:
            return

        obs = self.build_observation()
        action_index, action_id, confidence, logits = self._infer(obs)
        action_name = ACTION_NAMES.get(action_id, f"ACTION_{action_id}")

        proposal = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "action_index": action_index,
            "action_id": action_id,
            "action_name": action_name,
            "confidence": confidence,
            "needs_points": action_id in POINT_ACTIONS,
            "blocks_seen": len(self._extract_blocks()),
        }

        self.pending_action = proposal
        self.waiting_for_decision = self.require_confirmation

        self._publish_pending(proposal, logits)

        if self.require_confirmation:
            self.get_logger().info(
                f"PROPOSED action_id={action_id} ({action_name}) "
                f"conf={confidence:.3f}"
                + (
                    "  -> needs points; confirm with e.g. "
                    f"'confirm 250,200,90 270,220,45'"
                    if action_id in POINT_ACTIONS
                    else "  -> publish 'confirm' or 'cancel'"
                )
            )
        else:
            # Live mode: auto-run with no extra points.
            self._send_action(proposal, [])
            self.pending_action = None

    def _infer(self, obs: np.ndarray) -> Tuple[int, int, float, List[float]]:
        feeds = {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        outputs = self.session.run([self.output_name], feeds)
        logits = np.asarray(outputs[0]).reshape(-1).astype(np.float64)

        action_index = int(np.argmax(logits))
        confidence = float(self._softmax(logits)[action_index])

        if self.action_id_map is not None and action_index < len(self.action_id_map):
            action_id = int(self.action_id_map[action_index])
        else:
            action_id = action_index

        return action_index, action_id, confidence, [float(x) for x in logits]

    # ------------------------------------------------------------------ #
    # Observation builder  ***  ALIGN THIS WITH YOUR SIM'S OBS SPACE  ***
    # ------------------------------------------------------------------ #
    def build_observation(self) -> np.ndarray:
        """Build the policy input vector from the latest game state.

        IMPORTANT: this MUST match the observation space your sim trained with
        (feature order, normalization, dimension). The default layout below is a
        reasonable placeholder — edit it to match your exported policy.

        Default layout (length = _default_obs_length()):
          robot:  [x_norm, y_norm, sin(heading), cos(heading), pose_valid]      (5)
          blocks: for each of max_blocks nearest blocks:
                  [present, x_norm, y_norm, is_blue, is_red, in_goal]            (6 * max_blocks)

        The vector is fit (zero-padded / truncated) to self.obs_dim so inference
        always runs; a one-time warning is logged if the lengths differ.
        """
        feats: List[float] = []

        # Robot pose
        rx = ry = 0.0
        valid = 0.0
        sin_h = 0.0
        cos_h = 1.0
        if self.latest_odom and self.latest_odom.get("valid"):
            rx = float(self.latest_odom.get("x", 0.0))
            ry = float(self.latest_odom.get("y", 0.0))
            heading = math.radians(float(self.latest_odom.get("heading", 0.0)))
            sin_h, cos_h = math.sin(heading), math.cos(heading)
            valid = 1.0
        feats += [rx / self.field_size_cm, ry / self.field_size_cm, sin_h, cos_h, valid]

        # Blocks, nearest first
        blocks = self._extract_blocks()
        if valid:
            blocks.sort(key=lambda b: (b["x"] - rx) ** 2 + (b["y"] - ry) ** 2)

        for i in range(self.max_blocks):
            if i < len(blocks):
                b = blocks[i]
                name = b["class_name"].lower()
                feats += [
                    1.0,
                    b["x"] / self.field_size_cm,
                    b["y"] / self.field_size_cm,
                    1.0 if "blue" in name else 0.0,
                    1.0 if "red" in name else 0.0,
                    1.0 if "in_goal" in name else 0.0,
                ]
            else:
                feats += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return self._fit_obs(np.asarray(feats, dtype=np.float32))

    def _default_obs_length(self) -> int:
        return 5 + 6 * self.max_blocks

    def _fit_obs(self, vec: np.ndarray) -> np.ndarray:
        target = self.obs_dim if self.obs_dim > 0 else vec.shape[0]
        if vec.shape[0] == target:
            return vec
        if not self.obs_warned:
            self.get_logger().warning(
                f"Observation length {vec.shape[0]} != model expects {target}; "
                f"padding/truncating. Edit build_observation() to match your sim."
            )
            self.obs_warned = True
        if vec.shape[0] > target:
            return vec[:target]
        return np.concatenate([vec, np.zeros(target - vec.shape[0], dtype=np.float32)])

    def _extract_blocks(self) -> List[Dict[str, Any]]:
        """Pull block positions (field cm) from the latest mapped_objects packet."""
        out: List[Dict[str, Any]] = []
        if not self.latest_mapped:
            return out
        for obj in self.latest_mapped.get("objects", []) or []:
            try:
                out.append(
                    {
                        "class_name": str(obj.get("class_name", "")),
                        "x": float(obj.get("x_field_cm", 0.0)),
                        "y": float(obj.get("y_field_cm", 0.0)),
                    }
                )
            except (TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------ #
    # Action output
    # ------------------------------------------------------------------ #
    def _send_action(self, proposal: Dict[str, Any], points: List[Dict[str, float]]):
        action_id = int(proposal["action_id"])

        if action_id in POINT_ACTIONS and not points:
            self.get_logger().warning(
                f"Action {action_id} ({proposal['action_name']}) usually needs "
                f"points but none were given; sending with empty points."
            )

        command = {
            "type": "ACTION",
            "id": self._allocate_command_id(),
            "action_id": action_id,
            "points": points,
            "params": {
                "source": "policy",
                "confidence": proposal.get("confidence"),
                "action_index": proposal.get("action_index"),
            },
        }

        msg = String()
        msg.data = json.dumps(command)
        self.action_request_pub.publish(msg)

        self.get_logger().info(
            f"SENT to decision: action_id={action_id} "
            f"({proposal['action_name']}) points={len(points)}"
        )

    def _allocate_command_id(self) -> int:
        cid = self.next_command_id
        self.next_command_id += 1
        return cid

    def _publish_pending(self, proposal: Dict[str, Any], logits: List[float]):
        packet = dict(proposal)
        packet["logits"] = logits
        packet["instructions"] = (
            f"Publish to {self.decision_topic}: 'confirm' to run, 'cancel' to skip. "
            f"For point actions, append points: 'confirm x,y,heading x,y,heading'."
        )
        msg = String()
        msg.data = json.dumps(packet)
        self.pending_action_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    def _parse_decision(self, raw: str) -> Tuple[Optional[str], List[Dict[str, float]]]:
        text = (raw or "").strip()
        if not text:
            return None, []

        # JSON form: {"decision":"confirm","points":[{x,y,heading},...]}
        if text.startswith("{"):
            obj = self._safe_json(text) or {}
            word = str(obj.get("decision", obj.get("action", ""))).lower()
            points = self._normalize_points(obj.get("points", []))
            return self._decision_word(word), points

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
        # Accept "x,y,heading x,y,heading" or "x,y;x,y" etc.
        joined = " ".join(tokens).replace(";", " ")
        groups = [g for g in joined.split() if g.strip()]
        points = []
        for g in groups:
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
                        {
                            "x": float(p["x"]),
                            "y": float(p["y"]),
                            "heading": float(p.get("heading", 0.0)),
                        }
                    )
                elif isinstance(p, (list, tuple)):
                    out.append(
                        {
                            "x": float(p[0]),
                            "y": float(p[1]),
                            "heading": float(p[2]) if len(p) >= 3 else 0.0,
                        }
                    )
            except (KeyError, IndexError, ValueError, TypeError):
                continue
        return out

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        return e / s if s > 0 else np.ones_like(e) / len(e)

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
