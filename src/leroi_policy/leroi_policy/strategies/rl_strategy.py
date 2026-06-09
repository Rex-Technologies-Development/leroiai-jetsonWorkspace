"""RL strategy: run an exported RL policy (ONNX) and propose the argmax action.

This encapsulates the model session, observation building, and inference. It is
imported lazily by the factory so that scripted-only deployments do not require
onnxruntime/numpy to be installed.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from .actions import POINT_ACTIONS, action_name
from .base import ActionProposal, GameState, Strategy


class RLStrategy(Strategy):
    name = "rl"

    def __init__(self, cfg: Dict[str, Any], logger):
        self.log = logger
        self.model_path = os.path.expanduser(str(cfg.get("model_path", "")))
        self.intra_op_num_threads = int(cfg.get("intra_op_num_threads", 1))
        self.obs_dim_override = int(cfg.get("obs_dim_override", 0))
        self.max_blocks = int(cfg.get("max_blocks", 10))
        self.field_size_in = float(cfg.get("field_size_in", 144.0))
        self.onnx_providers = self._resolve_providers(str(cfg.get("onnx_providers", "CPUExecutionProvider")))
        self.action_id_map = self._parse_action_id_map(str(cfg.get("action_id_map", "")))

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.obs_dim: int = 0
        self.obs_warned = False
        self._load_model()

    # ------------------------------------------------------------------ #
    # Strategy interface
    # ------------------------------------------------------------------ #
    def propose(self, state: GameState) -> Optional[ActionProposal]:
        if self.session is None:
            return None

        obs = self.build_observation(state)
        feeds = {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        logits = np.asarray(self.session.run([self.output_name], feeds)[0]).reshape(-1).astype(np.float64)

        action_index = int(np.argmax(logits))
        confidence = float(self._softmax(logits)[action_index])

        if self.action_id_map is not None and action_index < len(self.action_id_map):
            action_id = int(self.action_id_map[action_index])
        else:
            action_id = action_index

        return ActionProposal(
            action_id=action_id,
            action_name=action_name(action_id),
            points=[],  # operator supplies points for point-actions at confirm time
            params={"action_index": action_index, "confidence": confidence},
            confidence=confidence,
            source="rl",
            info={
                "logits": [float(x) for x in logits],
                "needs_points": action_id in POINT_ACTIONS,
                "blocks_seen": len(state.blocks),
            },
        )

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "rl",
            "model_loaded": self.session is not None,
            "obs_dim": self.obs_dim,
            "providers": self.session.get_providers() if self.session else [],
        }

    # ------------------------------------------------------------------ #
    # Observation builder  ***  ALIGN THIS WITH YOUR SIM'S OBS SPACE  ***
    # ------------------------------------------------------------------ #
    def build_observation(self, state: GameState) -> np.ndarray:
        """Build the policy input vector from the latest game state.

        IMPORTANT: this MUST match the observation space your sim trained with
        (feature order, normalization, dimension). The default layout below is a
        reasonable placeholder — edit it to match your exported policy.

        Default layout (length = 5 + 6 * max_blocks):
          robot:  [x_norm, y_norm, sin(heading), cos(heading), pose_valid]   (5)
          blocks: for each of max_blocks nearest blocks:
                  [present, x_norm, y_norm, is_blue, is_red, in_goal]        (6 * max_blocks)
        """
        feats: List[float] = []

        rx = ry = 0.0
        valid = 0.0
        sin_h, cos_h = 0.0, 1.0
        if state.pose_valid:
            rx, ry = state.robot_xy
            heading = math.radians(state.robot_heading_deg)
            sin_h, cos_h = math.sin(heading), math.cos(heading)
            valid = 1.0
        feats += [rx / self.field_size_in, ry / self.field_size_in, sin_h, cos_h, valid]

        blocks = list(state.blocks)
        if valid:
            blocks.sort(key=lambda b: (b["x"] - rx) ** 2 + (b["y"] - ry) ** 2)

        for i in range(self.max_blocks):
            if i < len(blocks):
                b = blocks[i]
                name = b["class_name"].lower()
                feats += [
                    1.0,
                    b["x"] / self.field_size_in,
                    b["y"] / self.field_size_in,
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
            self.log.warning(
                f"Observation length {vec.shape[0]} != model expects {target}; "
                f"padding/truncating. Edit build_observation() to match your sim."
            )
            self.obs_warned = True
        if vec.shape[0] > target:
            return vec[:target]
        return np.concatenate([vec, np.zeros(target - vec.shape[0], dtype=np.float32)])

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            self.log.error(
                f"Policy model not found: {self.model_path}. "
                f"RL strategy will idle until a model is available."
            )
            return

        sess_options = ort.SessionOptions()
        if self.intra_op_num_threads > 0:
            sess_options.intra_op_num_threads = self.intra_op_num_threads

        try:
            self.session = ort.InferenceSession(
                self.model_path, sess_options=sess_options, providers=self.onnx_providers
            )
        except Exception as exc:
            self.log.error(f"Failed to load ONNX model: {exc}")
            self.session = None
            return

        self.log.info(f"ONNX running on providers: {self.session.get_providers()}")

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name

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
            self.obs_dim = self._default_obs_length()

        self.log.info(
            f"Loaded policy: input='{self.input_name}' shape={inp.shape} -> "
            f"output='{self.output_name}' shape={out.shape} | obs_dim={self.obs_dim}"
        )

    def _resolve_providers(self, raw: str) -> List[str]:
        available = ort.get_available_providers()
        requested = [p.strip() for p in raw.split(",") if p.strip()]
        resolved = [p for p in requested if p in available]
        dropped = [p for p in requested if p not in available]
        if dropped:
            self.log.warning(
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
            return [int(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, ValueError, TypeError):
            self.log.warning(f"Invalid action_id_map '{raw}'; using identity mapping.")
            return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        return e / s if s > 0 else np.ones_like(e) / len(e)
