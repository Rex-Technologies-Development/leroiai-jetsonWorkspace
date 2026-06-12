"""Route-forward strategy: propose the latest route planner suggestion as-is.

Minimal strategy for testing the route planner end to end:

    perception -> route planner (/perception/route_suggestion)
        -> THIS proposes it as a COLLECT_BLOCKS action (points cm -> inches)
        -> operator confirm/cancel gate (policy node)
        -> /decision/action_request -> decision -> /robot/command -> serial -> V5

It does no game logic; it just forwards whatever route the planner currently
suggests, so you can confirm/deny sending it to the V5 and verify the points on
the serial TX. The policy node's confirm gate means one command is sent per
confirm (it won't re-propose until you confirm or cancel the pending one).
"""

from typing import Any, Dict, Optional

from .actions import COLLECT_BLOCKS, action_name
from .base import ActionProposal, GameState, Strategy


class RouteForwardStrategy(Strategy):
    name = "route"

    def __init__(self, cfg: Dict[str, Any], logger):
        self.log = logger
        self.perception_to_robot_scale = float(cfg.get("perception_to_robot_scale", 1.0 / 2.54))
        self.max_points = int(cfg.get("max_collect_targets", 5))
        self._last_route_id: Optional[Any] = None

    def status(self) -> Dict[str, Any]:
        return {"mode": "route", "last_forwarded_route_id": self._last_route_id}

    def propose(self, state: GameState) -> Optional[ActionProposal]:
        route = state.route_suggestion
        if not route:
            return None

        raw = route.get("points") or []
        s = self.perception_to_robot_scale
        points = []
        for p in raw[: self.max_points]:
            try:
                points.append(
                    {"x": float(p["x"]) * s, "y": float(p["y"]) * s, "heading": float(p.get("heading", 0.0))}
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not points:
            return None  # empty route -> nothing to send

        self._last_route_id = route.get("id")
        action_id = int(route.get("action_id", COLLECT_BLOCKS))

        return ActionProposal(
            action_id=action_id,
            action_name=action_name(action_id),
            points=points,
            params={
                "source": "route_planner",
                "route_id": route.get("id"),
                "reason": route.get("reason"),
            },
            source="route_forward",
            info={
                "route_points_cm": raw,
                "candidate_count": route.get("candidate_count"),
            },
        )
