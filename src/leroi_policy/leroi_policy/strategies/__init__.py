"""Policy strategies: pluggable action selectors (RL model or scripted plan)."""

from typing import Any, Dict

from .actions import ACTION_NAMES, POINT_ACTIONS, action_name, extract_blocks
from .base import ActionProposal, GameState, Strategy

RL_MODES = {"rl", "model", "onnx", "policy", "learned"}
SCRIPTED_MODES = {"scripted", "defined", "rule", "rules", "fsm", "manual"}


def make_strategy(mode: str, cfg: Dict[str, Any], logger) -> Strategy:
    """Build the strategy for ``mode`` ('rl' or 'scripted').

    The RL strategy is imported lazily so scripted-only deployments don't need
    onnxruntime/numpy installed.
    """
    key = (mode or "rl").strip().lower()
    if key in RL_MODES:
        from .rl_strategy import RLStrategy

        return RLStrategy(cfg, logger)
    if key in SCRIPTED_MODES:
        from .scripted_strategy import ScriptedStrategy

        return ScriptedStrategy(cfg, logger)
    raise ValueError(
        f"Unknown strategy_mode '{mode}'. Use one of {sorted(RL_MODES)} "
        f"or {sorted(SCRIPTED_MODES)}."
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
