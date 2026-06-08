"""Scripted ("defined") strategy: a fixed, repeatable game plan, no learning.

The default plan loops:

    COLLECT_BLOCKS  ->  SCORE (alliance long goal)  ->  MOVE out to scan
        ->  if opponent blocks are visible:  DESCORE  ->  back to COLLECT
        ->  else:                            back to COLLECT

Every phase emits one action; the policy node only calls :meth:`propose` once the
previous action has been ACKed (decision node idle), so the machine advances one
step per completed action. All waypoints/ids are configurable — the defaults are
placeholders to tune against the real field.
"""

import json
import math
from typing import Any, Dict, List, Optional

from .actions import (
    COLLECT_BLOCKS,
    DEFAULT_DESCORE_BY_ALLIANCE,
    DEFAULT_SCORE_BY_ALLIANCE,
    MOVE_CHASSIS_ONLY,
    action_name,
)
from .base import ActionProposal, GameState, Strategy


class Phase:
    COLLECT = "collect"
    SCORE = "score"
    SCAN = "scan"
    EVALUATE = "evaluate"  # branch point: descore vs. collect again


class ScriptedStrategy(Strategy):
    name = "scripted"

    def __init__(self, cfg: Dict[str, Any], logger):
        self.log = logger

        self.alliance = str(cfg.get("alliance_color", "blue")).strip().lower()
        self.opponent_color = "red" if self.alliance == "blue" else "blue"

        self.score_action_id = int(
            cfg.get("score_action_id", DEFAULT_SCORE_BY_ALLIANCE.get(self.alliance, 3))
        )
        self.descore_action_id = int(
            cfg.get("descore_action_id", DEFAULT_DESCORE_BY_ALLIANCE.get(self.alliance, 13))
        )

        # Waypoints (field cm). JSON arrays of {x,y,heading} (or [x,y,heading]).
        self.scan_points = self._parse_points(
            cfg.get("scan_points_json", ""),
            default=[{"x": 182.0, "y": 182.0, "heading": 0.0}],
        )
        self.collect_fallback_points = self._parse_points(
            cfg.get("collect_fallback_points_json", ""),
            default=[{"x": 120.0, "y": 90.0, "heading": 0.0}],
        )

        self.max_collect_targets = int(cfg.get("max_collect_targets", 3))
        self.min_opponent_blocks_to_descore = int(cfg.get("min_opponent_blocks_to_descore", 1))

        self._phase = Phase.COLLECT
        self._loops = 0

    def reset(self) -> None:
        self._phase = Phase.COLLECT
        self._loops = 0

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "scripted",
            "phase": self._phase,
            "alliance": self.alliance,
            "loops_completed": self._loops,
            "score_action_id": self.score_action_id,
            "descore_action_id": self.descore_action_id,
        }

    # ------------------------------------------------------------------ #
    # Plan
    # ------------------------------------------------------------------ #
    def propose(self, state: GameState) -> Optional[ActionProposal]:
        phase = self._phase

        if phase == Phase.COLLECT:
            self._phase = Phase.SCORE
            return self._collect(state)

        if phase == Phase.SCORE:
            self._phase = Phase.SCAN
            return ActionProposal(
                action_id=self.score_action_id,
                action_name=action_name(self.score_action_id),
                points=[],
                params={"source": "scripted"},
                source="scripted:score",
            )

        if phase == Phase.SCAN:
            self._phase = Phase.EVALUATE
            return ActionProposal(
                action_id=MOVE_CHASSIS_ONLY,
                action_name=action_name(MOVE_CHASSIS_ONLY),
                points=[dict(p) for p in self.scan_points],
                params={"source": "scripted", "intent": "scan"},
                source="scripted:scan",
            )

        if phase == Phase.EVALUATE:
            # Decided against perception captured *after* the scan move finished.
            opp = self._opponent_blocks(state)
            if len(opp) >= self.min_opponent_blocks_to_descore:
                self._phase = Phase.COLLECT  # resume loop after descoring
                self._loops += 1
                return ActionProposal(
                    action_id=self.descore_action_id,
                    action_name=action_name(self.descore_action_id),
                    points=[],
                    params={"source": "scripted"},
                    source="scripted:descore",
                    info={"opponent_blocks_seen": len(opp)},
                )
            # No opponent blocks -> grab more and score again.
            self._phase = Phase.SCORE
            self._loops += 1
            return self._collect(state)

        # Unknown phase: re-home.
        self._phase = Phase.COLLECT
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _collect(self, state: GameState) -> ActionProposal:
        points = self._collect_targets(state)
        return ActionProposal(
            action_id=COLLECT_BLOCKS,
            action_name=action_name(COLLECT_BLOCKS),
            points=points,
            params={"source": "scripted"},
            source="scripted:collect",
            info={"used_fallback": not self._collectible_blocks(state)},
        )

    def _collect_targets(self, state: GameState) -> List[Dict[str, float]]:
        blocks = self._collectible_blocks(state)
        if state.robot_xy and blocks:
            rx, ry = state.robot_xy
            blocks.sort(key=lambda b: (b["x"] - rx) ** 2 + (b["y"] - ry) ** 2)
        points: List[Dict[str, float]] = []
        for b in blocks[: self.max_collect_targets]:
            points.append({"x": b["x"], "y": b["y"], "heading": self._heading_toward(state, b)})
        if not points:
            points = [dict(p) for p in self.collect_fallback_points]
        return points

    def _collectible_blocks(self, state: GameState) -> List[Dict[str, Any]]:
        """Blocks we should pick up: anything that is not an opponent block."""
        out = []
        for b in state.blocks:
            if self.opponent_color in b.get("class_name", "").lower():
                continue
            out.append(dict(b))
        return out

    def _opponent_blocks(self, state: GameState) -> List[Dict[str, Any]]:
        return [
            b for b in state.blocks if self.opponent_color in b.get("class_name", "").lower()
        ]

    def _heading_toward(self, state: GameState, block: Dict[str, float]) -> float:
        if not state.robot_xy:
            return 0.0
        rx, ry = state.robot_xy
        return math.degrees(math.atan2(block["y"] - ry, block["x"] - rx))

    def _parse_points(self, raw: Any, default: List[Dict[str, float]]) -> List[Dict[str, float]]:
        text = str(raw or "").strip()
        if not text:
            return [dict(p) for p in default]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            self.log.warning(f"Invalid points JSON '{text}'; using default.")
            return [dict(p) for p in default]
        out: List[Dict[str, float]] = []
        for p in parsed if isinstance(parsed, list) else []:
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
        return out or [dict(p) for p in default]
