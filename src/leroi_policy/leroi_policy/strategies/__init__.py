"""Policy strategies: pluggable action selectors (RL model or scripted plan)."""

from typing import Any, Dict

from .actions import ACTION_NAMES, POINT_ACTIONS, action_name, extract_blocks
from .base import ActionProposal, GameState, Strategy

RL_MODES = {"rl", "model", "onnx", "policy", "learned"}
SCRIPTED_MODES = {"scripted", "defined", "rule", "rules", "fsm", "manual"}
ROUTE_MODES = {"route", "route_forward", "forward_route", "route_planner"}
SCAN_SCORE_MODES = {"scan_score", "scan_action", "collect_scan_score"}


def make_strategy(mode: str, cfg: Dict[str, Any], logger) -> Strategy:
    """Build the strategy for ``mode`` ('rl', 'scripted', or 'route').

    The RL strategy is imported lazily so scripted/route-only deployments don't
    need onnxruntime/numpy installed.
    """
    key = (mode or "rl").strip().lower()
    if key in RL_MODES:
        from .rl_strategy import RLStrategy

        return RLStrategy(cfg, logger)
    if key in SCRIPTED_MODES:
        from .scripted_strategy import ScriptedStrategy

        return ScriptedStrategy(cfg, logger)
    if key in ROUTE_MODES:
        from .route_forward_strategy import RouteForwardStrategy

        return RouteForwardStrategy(cfg, logger)
    if key in SCAN_SCORE_MODES:
        from .scan_score_strategy import ScanScoreStrategy

        return ScanScoreStrategy(cfg, logger)
    raise ValueError(
        f"Unknown strategy_mode '{mode}'. Use one of {sorted(RL_MODES)}, "
        f"{sorted(SCRIPTED_MODES)}, {sorted(ROUTE_MODES)}, or {sorted(SCAN_SCORE_MODES)}."
    )


__all__ = [
    "ACTION_NAMES",
    "POINT_ACTIONS",
    "ActionProposal",
    "GameState",
    "Strategy",
    "action_name",
    "extract_blocks",
    "make_strategy",
]
