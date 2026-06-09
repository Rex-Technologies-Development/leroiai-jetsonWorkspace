"""Scripted ("defined") strategy: a fixed, repeatable game plan, no learning.

Plan (per the defined strategy):

    1. Move to a SCAN position near our parking zone and look at the goals.
    2. ASSESS the goal state from perception (our-color vs opponent-color blocks
       in the goal regions, plus how full they are) and branch:
         - mostly OPPONENT colors        -> DESCORE
         - mainly OUR colors             -> COLLECT more, then SCORE mid/low
         - neutral & ~empty              -> COLLECT more, then SCORE long goal
         - neutral & ~full               -> DESCORE
    3. After the descore/score finishes, return to the SCAN position.
    4. Repeat.

Every phase emits one action; the policy node only calls :meth:`propose` once the
previous action has been ACKed (decision node idle), so the machine advances one
step per completed action. Field-specific values (scan position, goal regions,
goal/score ids, thresholds) are all parameters — defaults are placeholders to
tune against the real field. Both alliances are supported; positions are supplied
independently per alliance.
"""

import json
import math
from typing import Any, Dict, List, Optional

from .actions import (
    COLLECT_BLOCKS,
    DEFAULT_DESCORE_BY_ALLIANCE,
    DEFAULT_SCORE_LONG_BY_ALLIANCE,
    DEFAULT_SCORE_MIDLOW_BY_ALLIANCE,
    MOVE_CHASSIS_ONLY,
    action_name,
)
from .base import ActionProposal, GameState, Strategy


class Phase:
    SCAN = "scan"            # move to the scan position near the parking zone
    ASSESS = "assess"        # branch on goal state (emits the chosen action)
    SCORE_MIDLOW = "score_midlow"
    SCORE_LONG = "score_long"


# Goal-state classifications.
MOSTLY_OPPONENT = "mostly_opponent"
MAINLY_OURS = "mainly_ours"
NEUTRAL_EMPTY = "neutral_empty"
NEUTRAL_FULL = "neutral_full"


class ScriptedStrategy(Strategy):
    name = "scripted"

    def __init__(self, cfg: Dict[str, Any], logger):
        self.log = logger

        self.alliance = str(cfg.get("alliance_color", "blue")).strip().lower()
        self.opponent_color = "red" if self.alliance == "blue" else "blue"

        self.score_long_action_id = self._id(
            cfg.get("score_long_action_id", 0),
            DEFAULT_SCORE_LONG_BY_ALLIANCE.get(self.alliance, 3),
        )
        self.score_midlow_action_id = self._id(
            cfg.get("score_midlow_action_id", 0),
            DEFAULT_SCORE_MIDLOW_BY_ALLIANCE.get(self.alliance, 7),
        )
        self.descore_action_id = self._id(
            cfg.get("descore_action_id", 0),
            DEFAULT_DESCORE_BY_ALLIANCE.get(self.alliance, 14),
        )

        # Alliance-specific positions (robot units = inches). Each alliance has
        # its own scan spot near its parking zone and its own fallback waypoints.
        self.scan_points = self._alliance_points(cfg, "scan_point", default=[
            {"x": 24.0, "y": 24.0, "heading": 45.0},
        ])
        self.collect_fallback_points = self._alliance_points(cfg, "collect_fallback", default=[
            {"x": 48.0, "y": 48.0, "heading": 0.0},
        ])

        self.max_collect_targets = int(cfg.get("max_collect_targets", 3))

        # Goal regions (perception field in) to assess. Each is a rectangle:
        # {"name","x_min","x_max","y_min","y_max"}. Empty -> assess ALL detected
        # blocks on the field (a rough proxy until goal regions are calibrated).
        self.goal_regions = self._parse_regions(cfg.get("goal_regions_json", ""))

        # Classification thresholds (counts of color blocks within goal regions).
        self.goal_color_margin = int(cfg.get("goal_color_margin", 1))  # |ours-opp| >= -> dominant
        self.goal_empty_max = int(cfg.get("goal_empty_max", 1))        # total <= -> "empty"
        self.goal_full_min = int(cfg.get("goal_full_min", 4))          # total >= -> "full"

        # Perception and robot/odom are both inches now, so this is identity by
        # default (kept as a knob in case a downstream frame ever differs).
        self.perception_to_robot_scale = float(cfg.get("perception_to_robot_scale", 1.0))

        # Keep this robot on its half of the field (robot units = inches).
        #   24in robot -> LEFT half:  zone_x_min_in=0,  zone_x_max_in=72
        #   15in robot -> RIGHT half: zone_x_min_in=72, zone_x_max_in=144
        # A little wander is allowed past each soft edge (zone_x_wander_in).
        # Waypoints past the hard edge (soft +/- wander) are clamped back in.
        # zone_x_max_in <= 0 disables the upper bound; zone_x_min_in <= 0 disables
        # the lower bound, so the default (0, 0) is "no zone limit".
        self.zone_x_min_in = float(cfg.get("zone_x_min_in", 0.0))
        self.zone_x_max_in = float(cfg.get("zone_x_max_in", 0.0))
        self.zone_x_wander_in = float(cfg.get("zone_x_wander_in", 12.0))

        # Prefer the route planner's COLLECT route (it already clusters, applies
        # wall margins, skips goals, and respects the pickup zone) over deriving
        # collect targets from raw blocks here.
        self.use_route_planner_for_collect = bool(cfg.get("use_route_planner_for_collect", True))

        self._phase = Phase.SCAN
        self._loops = 0
        self._last_assessment: Optional[str] = None

    def reset(self) -> None:
        self._phase = Phase.SCAN
        self._loops = 0
        self._last_assessment = None

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "scripted",
            "phase": self._phase,
            "alliance": self.alliance,
            "loops_completed": self._loops,
            "last_assessment": self._last_assessment,
            "score_long_action_id": self.score_long_action_id,
            "score_midlow_action_id": self.score_midlow_action_id,
            "descore_action_id": self.descore_action_id,
        }

    # ------------------------------------------------------------------ #
    # Plan
    # ------------------------------------------------------------------ #
    def propose(self, state: GameState) -> Optional[ActionProposal]:
        phase = self._phase

        if phase == Phase.SCAN:
            self._phase = Phase.ASSESS
            return ActionProposal(
                action_id=MOVE_CHASSIS_ONLY,
                action_name=action_name(MOVE_CHASSIS_ONLY),
                points=self._clamp_zone([dict(p) for p in self.scan_points]),
                params={"source": "scripted", "intent": "scan"},
                source="scripted:scan",
            )

        if phase == Phase.ASSESS:
            return self._assess_and_branch(state)

        if phase == Phase.SCORE_MIDLOW:
            self._phase = Phase.SCAN
            self._loops += 1
            return ActionProposal(
                action_id=self.score_midlow_action_id,
                action_name=action_name(self.score_midlow_action_id),
                points=[],
                params={"source": "scripted"},
                source="scripted:score_midlow",
            )

        if phase == Phase.SCORE_LONG:
            self._phase = Phase.SCAN
            self._loops += 1
            return ActionProposal(
                action_id=self.score_long_action_id,
                action_name=action_name(self.score_long_action_id),
                points=[],
                params={"source": "scripted"},
                source="scripted:score_long",
            )

        # Unknown phase: re-home to scan.
        self._phase = Phase.SCAN
        return None

    def _assess_and_branch(self, state: GameState) -> ActionProposal:
        """Decide the next action from the goal state captured after the scan."""
        classification, counts = self.assess_goals(state)
        self._last_assessment = classification

        if classification in (MOSTLY_OPPONENT, NEUTRAL_FULL):
            # Go remove opponent blocks; then back to scan.
            self._phase = Phase.SCAN
            self._loops += 1
            return ActionProposal(
                action_id=self.descore_action_id,
                action_name=action_name(self.descore_action_id),
                points=[],
                params={"source": "scripted"},
                source="scripted:descore",
                info={"assessment": classification, **counts},
            )

        # Otherwise collect more first; the follow-up score depends on the state.
        if classification == MAINLY_OURS:
            self._phase = Phase.SCORE_MIDLOW
        else:  # NEUTRAL_EMPTY (and any fallthrough)
            self._phase = Phase.SCORE_LONG

        return self._collect(state, classification, counts)

    # ------------------------------------------------------------------ #
    # Goal-state assessment
    # ------------------------------------------------------------------ #
    def assess_goals(self, state: GameState):
        """Classify the goals from perception.

        Counts our-color vs opponent-color blocks that fall inside the configured
        goal regions (or all detected blocks if no regions are set). Returns
        (classification, counts_dict).
        """
        ours = opp = 0
        for b in state.blocks:
            if not self._in_goal_regions(b):
                continue
            name = str(b.get("class_name", "")).lower()
            if self.alliance in name:
                ours += 1
            elif self.opponent_color in name:
                opp += 1

        total = ours + opp
        counts = {"ours_in_goals": ours, "opp_in_goals": opp, "total_in_goals": total}

        if opp - ours >= self.goal_color_margin:
            return MOSTLY_OPPONENT, counts
        if ours - opp >= self.goal_color_margin:
            return MAINLY_OURS, counts
        # Roughly balanced -> decide by fullness.
        if total >= self.goal_full_min:
            return NEUTRAL_FULL, counts
        return NEUTRAL_EMPTY, counts

    def _in_goal_regions(self, block: Dict[str, Any]) -> bool:
        if not self.goal_regions:
            return True  # no regions configured -> assess everything seen
        try:
            x = float(block.get("x", 0.0))
            y = float(block.get("y", 0.0))
        except (TypeError, ValueError):
            return False
        for r in self.goal_regions:
            if r["x_min"] <= x <= r["x_max"] and r["y_min"] <= y <= r["y_max"]:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Collect helpers
    # ------------------------------------------------------------------ #
    def _collect(self, state: GameState, classification: str, counts: Dict[str, Any]) -> ActionProposal:
        points = self._collect_targets(state)
        return ActionProposal(
            action_id=COLLECT_BLOCKS,
            action_name=action_name(COLLECT_BLOCKS),
            points=points,
            params={"source": "scripted"},
            source="scripted:collect",
            info={"assessment": classification, "used_fallback": not self._collectible_blocks(state), **counts},
        )

    def _collect_targets(self, state: GameState) -> List[Dict[str, float]]:
        # 1) Prefer the route planner's suggestion (already alliance/zone-aware).
        route_pts = self._route_suggestion_points(state)
        if route_pts:
            return self._clamp_zone(route_pts[: self.max_collect_targets])

        # 2) Else derive from raw detected blocks, nearest first.
        blocks = self._collectible_blocks(state)
        if state.robot_xy and blocks:
            rx, ry = state.robot_xy
            blocks.sort(key=lambda b: (b["x"] - rx) ** 2 + (b["y"] - ry) ** 2)
        points: List[Dict[str, float]] = []
        for b in blocks[: self.max_collect_targets]:
            points.append({"x": b["x"], "y": b["y"], "heading": self._heading_toward(state, b)})

        # 3) Else fixed fallback waypoints (already in robot units).
        if not points:
            points = [dict(p) for p in self.collect_fallback_points]
        return self._clamp_zone(points)

    def _route_suggestion_points(self, state: GameState) -> List[Dict[str, float]]:
        """Points from the latest route planner suggestion (already inches)."""
        if not self.use_route_planner_for_collect or not state.route_suggestion:
            return []
        raw = state.route_suggestion.get("points") or []
        s = self.perception_to_robot_scale
        out: List[Dict[str, float]] = []
        for p in raw:
            try:
                out.append(
                    {
                        "x": float(p["x"]) * s,
                        "y": float(p["y"]) * s,
                        "heading": float(p.get("heading", 0.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _clamp_zone(self, points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Clamp waypoint x to the robot's half of the field.

        Upper bound active when zone_x_max_in > 0 (keeps the LEFT-side robot off
        the right half); lower bound active when zone_x_min_in > 0 (keeps the
        RIGHT-side robot off the left half). Each edge allows zone_x_wander_in of
        slack before the hard clamp.
        """
        has_max = self.zone_x_max_in > 0
        has_min = self.zone_x_min_in > 0
        if not has_max and not has_min:
            return points
        hard_max = self.zone_x_max_in + self.zone_x_wander_in
        hard_min = self.zone_x_min_in - self.zone_x_wander_in
        clamped = []
        for p in points:
            q = dict(p)
            if has_max and q["x"] > hard_max:
                q["x"] = hard_max
            if has_min and q["x"] < hard_min:
                q["x"] = hard_min
            clamped.append(q)
        return clamped

    def _blocks_robot_frame(self, state: GameState) -> List[Dict[str, Any]]:
        """Blocks with positions in robot inches (perception is already inches)."""
        s = self.perception_to_robot_scale
        out = []
        for b in state.blocks:
            try:
                out.append(
                    {
                        "class_name": str(b.get("class_name", "")),
                        "x": float(b.get("x", 0.0)) * s,
                        "y": float(b.get("y", 0.0)) * s,
                    }
                )
            except (TypeError, ValueError):
                continue
        return out

    def _collectible_blocks(self, state: GameState) -> List[Dict[str, Any]]:
        """Blocks we should pick up (robot units): anything not an opponent block."""
        return [
            b
            for b in self._blocks_robot_frame(state)
            if self.opponent_color not in b["class_name"].lower()
        ]

    def _heading_toward(self, state: GameState, block: Dict[str, float]) -> float:
        if not state.robot_xy:
            return 0.0
        rx, ry = state.robot_xy
        return math.degrees(math.atan2(block["y"] - ry, block["x"] - rx))

    # ------------------------------------------------------------------ #
    # Param parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _id(value: Any, default: int) -> int:
        v = int(value)
        return v if v > 0 else int(default)

    def _alliance_points(self, cfg: Dict[str, Any], key: str, default: List[Dict[str, float]]):
        """Pick the alliance-specific `<key>_<alliance>_json`, else the default."""
        return self._parse_points(cfg.get(f"{key}_{self.alliance}_json", ""), default=default)

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

    def _parse_regions(self, raw: Any) -> List[Dict[str, float]]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            self.log.warning(f"Invalid goal_regions JSON '{text}'; assessing all blocks.")
            return []
        out: List[Dict[str, float]] = []
        for r in parsed if isinstance(parsed, list) else []:
            try:
                out.append(
                    {
                        "name": str(r.get("name", f"goal{len(out)}")),
                        "x_min": float(r["x_min"]),
                        "x_max": float(r["x_max"]),
                        "y_min": float(r["y_min"]),
                        "y_max": float(r["y_max"]),
                    }
                )
            except (KeyError, ValueError, TypeError):
                self.log.warning(f"Skipping invalid goal region: {r}")
        return out
