"""Shared types for policy strategies.

A *strategy* is anything that, given the current game state, decides the next
robot action. Two implementations live alongside each other:

  - RLStrategy       : runs an exported RL model (ONNX) and picks argmax action.
  - ScriptedStrategy : executes a fixed, repeatable game plan (no learning).

Both produce an :class:`ActionProposal`, which the policy node funnels through
the same operator confirm/cancel gate and on to ``/decision/action_request``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GameState:
    """Snapshot of everything a strategy needs to choose the next action.

    The policy node builds this once per tick from the latest cached perception
    and odometry packets, so strategies never touch ROS directly.
    """

    odom: Optional[Dict[str, Any]] = None            # parsed /v5/odom packet
    mapped_objects: Optional[Dict[str, Any]] = None  # parsed mapped_objects packet
    detections: Optional[Dict[str, Any]] = None      # parsed detections packet
    blocks: List[Dict[str, Any]] = field(default_factory=list)  # extracted, field cm
    decision_busy: bool = False                      # decision node waiting for an ACK
    now_ns: int = 0

    @property
    def pose_valid(self) -> bool:
        return bool(self.odom and self.odom.get("valid"))

    @property
    def robot_xy(self) -> Optional[Tuple[float, float]]:
        if self.pose_valid:
            return float(self.odom.get("x", 0.0)), float(self.odom.get("y", 0.0))
        return None

    @property
    def robot_heading_deg(self) -> float:
        if self.odom:
            return float(self.odom.get("heading", 0.0))
        return 0.0


@dataclass
class ActionProposal:
    """A proposed action, agnostic of how it was chosen (RL or scripted)."""

    action_id: int
    action_name: str = ""
    points: List[Dict[str, float]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    source: str = ""                                 # e.g. "rl", "scripted:collect"
    info: Dict[str, Any] = field(default_factory=dict)  # debug extras (logits, phase)


class Strategy(ABC):
    """Common interface for anything that proposes the next robot action."""

    name: str = "base"

    @abstractmethod
    def propose(self, state: GameState) -> Optional[ActionProposal]:
        """Return the next action to attempt, or ``None`` to idle this tick.

        Called by the policy node only when it is ready to issue a new action
        (no action pending confirmation, and the decision node is not waiting on
        an ACK), so a scripted strategy can safely advance its plan one step per
        call.
        """

    def reset(self) -> None:
        """Reset internal state (e.g. when restarting a scripted plan)."""

    def status(self) -> Dict[str, Any]:
        """Optional strategy-specific status for /policy/status. Override me."""
        return {}
