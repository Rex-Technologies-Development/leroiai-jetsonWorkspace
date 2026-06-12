"""Scan-score strategy: collect -> scan & face goal -> score / descore.

Follow-up to the route-planner test. Flow (one action per ACK, each held behind the
policy node's confirm/cancel gate):

    1. COLLECT — drive the route planner's pickup route (points cm -> inches).
    2. SCAN    — move to the scan point near the parking zone, facing the goal.
    3. DECIDE based on the long goal, then loop back to COLLECT.

Goal state uses the YOLO *class* (not x/y): blocks detected as "in a goal" are counted
by color via class-name keywords. Two decide modes:

  - fallback_descore_then_score=True (default): always DESCORE then SCORE. Descore is
    harmless even with no opponent blocks, so this is the safe/simple default.
  - else (class assessment): majority opponent in-goal -> DESCORE; empty/partial -> SCORE.
"""

from typing import Any, Dict, Optional, Tuple

from .actions import COLLECT_BLOCKS, MOVE_CHASSIS_ONLY, action_name
from .base import ActionProposal, GameState
from .scripted_strategy import ScriptedStrategy


class ScanScoreStrategy(ScriptedStrategy):
    name = "scan_score"

    def __init__(self, cfg: Dict[str, Any], logger):
        super().__init__(cfg, logger)
        self.long_goal_capacity = int(cfg.get("long_goal_capacity", 14))
        # Safe default: descore (harmless) then score, no assessment needed.
        self.fallback_descore_then_score = bool(cfg.get("fallback_descore_then_score", True))
        # YOLO class-name substrings that mean "this block is in a goal".
        kw = str(cfg.get("in_goal_keywords", "goal,in_goal,scored")).split(",")
        self.in_goal_keywords = [k.strip().lower() for k in kw if k.strip()]
        self._phase = "collect"

    def reset(self) -> None:
        self._phase = "collect"
        self._loops = 0
        self._last_assessment = None

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "scan_score",
            "phase": self._phase,
            "alliance": self.alliance,
            "loops_completed": self._loops,
            "last_assessment": self._last_assessment,
            "fallback_descore_then_score": self.fallback_descore_then_score,
            "score_long_action_id": self.score_long_action_id,
            "descore_action_id": self.descore_action_id,
        }

    def propose(self, state: GameState) -> Optional[ActionProposal]:
        if self._phase == "collect":
            self._phase = "scan"
            return ActionProposal(
                action_id=COLLECT_BLOCKS,
                action_name=action_name(COLLECT_BLOCKS),
                points=self._collect_targets(state),
                params={"source": "scan_score"},
                source="scan_score:collect",
                info={"used_route": bool(self._route_suggestion_points(state))},
            )

        if self._phase == "scan":
            self._phase = "decide"
            return ActionProposal(
                action_id=MOVE_CHASSIS_ONLY,
                action_name=action_name(MOVE_CHASSIS_ONLY),
                points=self._clamp_zone([dict(p) for p in self.scan_points]),
                params={"source": "scan_score", "intent": "scan_face_goal"},
                source="scan_score:scan",
            )

        if self._phase == "decide":
            ours, opp, in_goal = self._assess_by_class(state)
            counts = {"ours_in_goal": ours, "opp_in_goal": opp, "in_goal_total": in_goal}

            # Fallback: descore (harmless) then score, regardless of what we see.
            if self.fallback_descore_then_score:
                self._phase = "score_after_descore"
                self._last_assessment = f"fallback descore-then-score ({counts})"
                return self._descore(counts)

            # Class assessment: majority opponent -> descore, else score.
            self._phase = "collect"
            self._loops += 1
            if opp > ours:
                self._last_assessment = f"majority_opponent {counts} -> DESCORE"
                return self._descore(counts)
            self._last_assessment = f"{'empty' if in_goal == 0 else 'ours/partial'} {counts} -> SCORE"
            return self._score(counts)

        # score_after_descore (fallback's second step)
        self._phase = "collect"
        self._loops += 1
        self._last_assessment = "fallback: score after descore"
        return self._score({})

    # ------------------------------------------------------------------ #
    def _assess_by_class(self, state: GameState) -> Tuple[int, int, int]:
        """Count in-goal blocks by color using YOLO class names (no x/y)."""
        ours = opp = in_goal = 0
        for b in state.blocks:
            name = str(b.get("class_name", "")).lower()
            if not any(k in name for k in self.in_goal_keywords):
                continue
            in_goal += 1
            if self.alliance in name:
                ours += 1
            elif self.opponent_color in name:
                opp += 1
        return ours, opp, in_goal

    def _descore(self, info: Dict[str, Any]) -> ActionProposal:
        return ActionProposal(
            action_id=self.descore_action_id,
            action_name=action_name(self.descore_action_id),
            points=[],
            params={"source": "scan_score"},
            source="scan_score:descore",
            info=info,
        )

    def _score(self, info: Dict[str, Any]) -> ActionProposal:
        return ActionProposal(
            action_id=self.score_long_action_id,
            action_name=action_name(self.score_long_action_id),
            points=[],
            params={"source": "scan_score"},
            source="scan_score:score_long",
            info=info,
        )
