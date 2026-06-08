"""Action-id vocabulary shared by policy strategies.

This mirrors ``leroi_decision/decision_node.py``'s ``ACTION_NAMES``. Kept local
so this package does not import leroi_decision. If you change one, change both.
"""

from typing import Any, Dict, List

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
    24: "RESET_GYRO",
}

# Named ids for readability in strategies.
STOP = 0
MOVE_CHASSIS_ONLY = 1
COLLECT_BLOCKS = 2
RESET_GYRO = 24

# Actions that consume target points. Everything else auto-navigates on the V5.
POINT_ACTIONS = {MOVE_CHASSIS_ONLY, COLLECT_BLOCKS}

# Logical groups.
SCORE_ACTION_IDS = set(range(3, 11))     # 3..10
DESCORE_ACTION_IDS = set(range(11, 19))  # 11..18

# Sensible alliance-aware defaults for the scripted strategy. Tune to your field.
DEFAULT_SCORE_LONG_BY_ALLIANCE = {
    "blue": 3,   # SCORE_LEFT_LONG_GOAL_ALLIANCE
    "red": 5,    # SCORE_RIGHT_LONG_GOAL_ALLIANCE
}
DEFAULT_SCORE_MIDLOW_BY_ALLIANCE = {
    "blue": 7,   # SCORE_MID_LEFT   (placeholder; tune mid vs low per side)
    "red": 8,    # SCORE_MID_RIGHT
}
# Wall-side descore (operator-selected).
DEFAULT_DESCORE_BY_ALLIANCE = {
    "blue": 14,  # DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL
    "red": 18,   # DESCORE_RIGHT_LONG_GOAL_OPPONENT_WALL
}


def action_name(action_id: int) -> str:
    return ACTION_NAMES.get(int(action_id), f"ACTION_{action_id}")


def extract_blocks(mapped_objects: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull block positions (field cm) from a mapped_objects packet."""
    out: List[Dict[str, Any]] = []
    if not mapped_objects:
        return out
    for obj in mapped_objects.get("objects", []) or []:
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
