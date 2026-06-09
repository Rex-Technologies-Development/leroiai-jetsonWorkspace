import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


ACTION_ID_COLLECT_BLOCKS = 2
ACTION_NAME_COLLECT_BLOCKS = "COLLECT_BLOCKS"


class RoutePlannerNode(Node):
    """
    Route planner for selecting block pickup target points.

    Inputs:
      /perception/oak_front/detections
        Expected JSON containing:
          {
            "robot_pose_valid": true,
            "robot_x": ...,
            "robot_y": ...,
            "robot_heading_deg": ...,
            "detections": [
              {
                "class_name": "blue_block",
                "confidence": 0.92,
                "x_field_in": 100.0,
                "y_field_in": 80.0,
                "z_in": 120.0
              }
            ]
          }

      /v5/odom
        Used as fallback/current robot pose.

    Outputs:
      /perception/route_suggestion
        {
          "type": "ROUTE_SUGGESTION",
          "id": ...,
          "action_id": 2,
          "action_name": "COLLECT_BLOCKS",
          "alliance_color": "blue",
          "target_color": "blue",
          "opponent_color": "red",
          "points": [
            {"x": 100.0, "y": 80.0, "heading": 35.0}
          ],
          "params": {...}
        }

      /perception/route_status
        Debug/status info.

    Coordinate convention:
      field coordinates in inches
      (0, 0) = bottom-left
      +X = right
      +Y = up/forward
      heading 0 deg = +Y
    """

    def __init__(self):
        super().__init__("route_planner_node")

        # Topics
        self.declare_parameter(
            "detections_topic",
            "/perception/oak_front/detections",
        )
        self.declare_parameter("odom_topic", "/v5/odom")
        self.declare_parameter(
            "route_suggestion_topic",
            "/perception/route_suggestion",
        )
        self.declare_parameter(
            "route_status_topic",
            "/perception/route_status",
        )

        # Alliance
        self.declare_parameter("alliance_color", "blue")

        # Planner timing
        self.declare_parameter("route_rate_hz", 3.0)
        self.declare_parameter("stale_detection_timeout_sec", 1.0)
        self.declare_parameter("stale_odom_timeout_sec", 1.0)
        # When False, plan from whatever pose the detection packet carries (e.g.
        # the perception node's assumed-origin pose) without requiring fresh
        # /v5/odom. Lets routes appear before the V5 brain streams odometry.
        self.declare_parameter("require_fresh_odom", True)

        # Field bounds
        self.declare_parameter("field_min_x_in", 0.0)
        self.declare_parameter("field_max_x_in", 144.0)
        self.declare_parameter("field_min_y_in", 0.0)
        self.declare_parameter("field_max_y_in", 144.0)

        # Route limits
        self.declare_parameter("max_route_points", 5)
        self.declare_parameter("max_carry_blocks", 5)
        self.declare_parameter("already_held_blocks", 0)

        # Pickup zone limit (field in). Restricts which blocks this robot will be
        # routed to, independent of the field bounds used for geometry. For the
        # 24in robot we keep it on the LEFT half (small wander allowed) so it does
        # not impede the 15in robot on the right. Defaults = no restriction.
        self.declare_parameter("pickup_x_min_in", -1.0e9)
        self.declare_parameter("pickup_x_max_in", 1.0e9)

        # Scoring / filtering
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("wall_margin_in", 17.72)
        self.declare_parameter("cluster_dist_in", 35.43)
        self.declare_parameter("cluster_pickup_dist_in", 14.96)
        # Cutoff on value = score / (dist_in + DIST_SOFTEN_IN). Scaled up by 2.54
        # from the old centimeter cutoff (0.015) since value is per-inch now.
        self.declare_parameter("min_route_value", 0.0381)

        # No-pickup zones around field structures (field inches).
        self.declare_parameter("skip_near_goals", True)
        # Center X: 22in-diameter cross of two 5in-thick diagonal bars.
        self.declare_parameter("center_x_radius_in", 11.0)
        self.declare_parameter("center_x_thickness_in", 5.0)
        # LEFT long goal as an explicit rectangle: x in [min,max], y in [min,max].
        # Measured corners (21,48)-(27,96). The RIGHT long goal is the mirror of
        # this across the field center-x (field_max_x - [max,min]) -> [117,123].
        self.declare_parameter("long_goal_x_min_in", 21.0)
        self.declare_parameter("long_goal_x_max_in", 27.0)
        self.declare_parameter("long_goal_y_min_in", 48.0)
        self.declare_parameter("long_goal_y_max_in", 96.0)
        # Safety padding grown around every goal rectangle before skipping a block.
        self.declare_parameter("goal_skip_pad_in", 0.0)
        self.declare_parameter("top_bottom_goal_x_min_in", 47.24)
        self.declare_parameter("top_bottom_goal_x_max_in", 96.85)
        self.declare_parameter("top_bottom_goal_y_margin_in", 21.65)

        # Behavior
        self.declare_parameter("publish_empty_route", True)
        self.declare_parameter("debug_log_routes", True)

        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.route_suggestion_topic = str(
            self.get_parameter("route_suggestion_topic").value
        )
        self.route_status_topic = str(self.get_parameter("route_status_topic").value)

        self.alliance_color = (
            str(self.get_parameter("alliance_color").value).lower().strip()
        )
        if self.alliance_color not in ("red", "blue"):
            self.get_logger().warning(
                f"Invalid alliance_color={self.alliance_color}. Defaulting to blue."
            )
            self.alliance_color = "blue"

        self.target_color = self.alliance_color
        self.opponent_color = "red" if self.alliance_color == "blue" else "blue"

        self.route_rate_hz = float(self.get_parameter("route_rate_hz").value)
        self.stale_detection_timeout_sec = float(
            self.get_parameter("stale_detection_timeout_sec").value
        )
        self.stale_odom_timeout_sec = float(
            self.get_parameter("stale_odom_timeout_sec").value
        )
        self.require_fresh_odom = bool(
            self.get_parameter("require_fresh_odom").value
        )

        self.field_min_x_in = float(self.get_parameter("field_min_x_in").value)
        self.field_max_x_in = float(self.get_parameter("field_max_x_in").value)
        self.field_min_y_in = float(self.get_parameter("field_min_y_in").value)
        self.field_max_y_in = float(self.get_parameter("field_max_y_in").value)

        self.max_route_points = int(self.get_parameter("max_route_points").value)
        self.max_carry_blocks = int(self.get_parameter("max_carry_blocks").value)
        self.already_held_blocks = int(
            self.get_parameter("already_held_blocks").value
        )

        self.pickup_x_min_in = float(self.get_parameter("pickup_x_min_in").value)
        self.pickup_x_max_in = float(self.get_parameter("pickup_x_max_in").value)

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.wall_margin_in = float(self.get_parameter("wall_margin_in").value)
        self.cluster_dist_in = float(self.get_parameter("cluster_dist_in").value)
        self.cluster_pickup_dist_in = float(
            self.get_parameter("cluster_pickup_dist_in").value
        )
        self.min_route_value = float(self.get_parameter("min_route_value").value)

        self.skip_near_goals = bool(self.get_parameter("skip_near_goals").value)
        self.center_x_radius_in = float(self.get_parameter("center_x_radius_in").value)
        self.center_x_thickness_in = float(
            self.get_parameter("center_x_thickness_in").value
        )
        self.long_goal_x_min_in = float(self.get_parameter("long_goal_x_min_in").value)
        self.long_goal_x_max_in = float(self.get_parameter("long_goal_x_max_in").value)
        self.long_goal_y_min_in = float(self.get_parameter("long_goal_y_min_in").value)
        self.long_goal_y_max_in = float(self.get_parameter("long_goal_y_max_in").value)
        self.goal_skip_pad_in = float(self.get_parameter("goal_skip_pad_in").value)
        self.top_bottom_goal_x_min_in = float(
            self.get_parameter("top_bottom_goal_x_min_in").value
        )
        self.top_bottom_goal_x_max_in = float(
            self.get_parameter("top_bottom_goal_x_max_in").value
        )
        self.top_bottom_goal_y_margin_in = float(
            self.get_parameter("top_bottom_goal_y_margin_in").value
        )

        self.publish_empty_route = bool(
            self.get_parameter("publish_empty_route").value
        )
        self.debug_log_routes = bool(self.get_parameter("debug_log_routes").value)

        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_detection_time = 0.0
        self.latest_detection_stamp_ns = 0

        self.robot_pose_valid = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading_deg = 0.0
        self.latest_odom_time = 0.0
        self.latest_odom_stamp_ns = 0

        self.route_sequence = 0

        self.route_pub = self.create_publisher(
            String,
            self.route_suggestion_topic,
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            self.route_status_topic,
            10,
        )

        self.detections_sub = self.create_subscription(
            String,
            self.detections_topic,
            self.detections_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            String,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.route_rate_hz,
            self.route_timer_callback,
        )

        self.get_logger().info("Route planner node started.")
        self.get_logger().info(f"Alliance color: {self.alliance_color}")
        self.get_logger().info(f"Target color: {self.target_color}")
        self.get_logger().info(f"Opponent color: {self.opponent_color}")
        self.get_logger().info(f"Detections topic: {self.detections_topic}")
        self.get_logger().info(f"Route suggestion topic: {self.route_suggestion_topic}")

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------

    def detections_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        detections = data.get("detections", [])
        if not isinstance(detections, list):
            return

        self.latest_detections = detections
        self.latest_detection_time = time.time()
        self.latest_detection_stamp_ns = int(data.get("t", 0))

        # Use detection packet pose if available.
        if bool(data.get("robot_pose_valid", False)):
            try:
                self.robot_x = float(data.get("robot_x", self.robot_x))
                self.robot_y = float(data.get("robot_y", self.robot_y))
                self.robot_heading_deg = float(
                    data.get("robot_heading_deg", self.robot_heading_deg)
                )
                self.robot_pose_valid = True
            except (ValueError, TypeError):
                pass

    def odom_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not bool(data.get("valid", True)):
            return

        try:
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            heading = float(data.get("heading", 0.0))
        except (ValueError, TypeError):
            return

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
            return

        self.robot_x = x
        self.robot_y = y
        self.robot_heading_deg = heading
        self.robot_pose_valid = True
        self.latest_odom_time = time.time()
        self.latest_odom_stamp_ns = int(data.get("stamp_ns", 0))

    # -------------------------------------------------------------------------
    # Planner
    # -------------------------------------------------------------------------

    def route_timer_callback(self):
        now = time.time()

        detection_age = (
            None
            if self.latest_detection_time <= 0.0
            else now - self.latest_detection_time
        )
        odom_age = None if self.latest_odom_time <= 0.0 else now - self.latest_odom_time

        detections_stale = (
            detection_age is None
            or detection_age > self.stale_detection_timeout_sec
        )
        odom_stale = odom_age is None or odom_age > self.stale_odom_timeout_sec

        odom_blocked = (not self.robot_pose_valid) or (
            self.require_fresh_odom and odom_stale
        )
        if odom_blocked:
            self.publish_status(
                route=[],
                candidates=[],
                reason="NO_VALID_ODOM",
                detection_age=detection_age,
                odom_age=odom_age,
            )
            if self.publish_empty_route:
                self.publish_route([], [], "NO_VALID_ODOM")
            return

        if detections_stale:
            self.publish_status(
                route=[],
                candidates=[],
                reason="NO_RECENT_DETECTIONS",
                detection_age=detection_age,
                odom_age=odom_age,
            )
            if self.publish_empty_route:
                self.publish_route([], [], "NO_RECENT_DETECTIONS")
            return

        candidates = self.build_candidates(self.latest_detections)
        route = self.compute_route(candidates)

        reason = "OK" if route else "NO_ROUTE"

        self.publish_route(route, candidates, reason)
        self.publish_status(
            route=route,
            candidates=candidates,
            reason=reason,
            detection_age=detection_age,
            odom_age=odom_age,
        )

        if self.debug_log_routes and route:
            self.get_logger().info(
                f"Route suggestion: alliance={self.alliance_color} "
                f"points={len(route)} "
                f"first=({route[0]['x']:.1f},{route[0]['y']:.1f})"
            )

    def build_candidates(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []

        for i, det in enumerate(detections):
            if not isinstance(det, dict):
                continue

            class_name = str(det.get("class_name", det.get("class", "")))
            color = self.detect_color_from_class(class_name)

            if color != self.target_color:
                continue

            confidence = self.safe_float(det.get("confidence", det.get("conf", 0.0)))

            if confidence < self.min_confidence:
                continue

            x_field = det.get("x_field_in")
            y_field = det.get("y_field_in")

            if x_field is None or y_field is None:
                continue

            x = self.safe_float(x_field)
            y = self.safe_float(y_field)

            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            if not self.in_field_bounds(x, y):
                continue

            # Keep this robot inside its pickup zone (e.g. left half for the 24in).
            if not (self.pickup_x_min_in <= x <= self.pickup_x_max_in):
                continue

            if self.near_wall_count(x, y) >= 2:
                continue

            if self.skip_near_goals and self.near_any_goal(x, y):
                continue

            same_nearby, opp_nearby = self.cluster_counts(
                x=x,
                y=y,
                detections=detections,
            )

            score = self.score_candidate(
                x=x,
                y=y,
                same_nearby=same_nearby,
                opp_nearby=opp_nearby,
            )

            if score <= 0.0:
                continue

            candidates.append(
                {
                    "source_index": i,
                    "class_name": class_name,
                    "color": color,
                    "confidence": confidence,
                    "x": x,
                    "y": y,
                    "x_in": det.get("x_in"),
                    "z_in": det.get("z_in"),
                    "score": score,
                    "same_nearby": same_nearby,
                    "opp_nearby": opp_nearby,
                }
            )

        return candidates

    def compute_route(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        slots_left = min(
            self.max_carry_blocks - self.already_held_blocks,
            self.max_route_points,
        )

        if slots_left <= 0 or not candidates:
            return []

        clusters = self.build_pickup_clusters(candidates)

        route = []
        used_cluster_indices = set()
        current_x = self.robot_x
        current_y = self.robot_y

        while len(route) < slots_left:
            best_idx = None
            best_value = -1.0

            for idx, cluster in enumerate(clusters):
                if idx in used_cluster_indices:
                    continue

                dist = math.hypot(cluster["x"] - current_x, cluster["y"] - current_y)
                # Distance-softening constant (inches).
                value = cluster["score"] / (dist + 19.69)

                if value > best_value:
                    best_value = value
                    best_idx = idx

            if best_idx is None:
                break

            if best_value < self.min_route_value:
                break

            cluster = clusters[best_idx]
            heading = self.heading_from_to(
                from_x=current_x,
                from_y=current_y,
                to_x=cluster["x"],
                to_y=cluster["y"],
            )

            route.append(
                {
                    "x": round(cluster["x"], 2),
                    "y": round(cluster["y"], 2),
                    "heading": round(heading, 2),
                    "score": round(cluster["score"], 3),
                    "source_count": len(cluster["source_indices"]),
                    "source_indices": cluster["source_indices"],
                    "class_name": cluster["class_name"],
                }
            )

            used_cluster_indices.add(best_idx)
            current_x = cluster["x"]
            current_y = cluster["y"]

        return route

    def build_pickup_clusters(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        unvisited = list(candidates)
        clusters = []

        while unvisited:
            cluster_entries = [unvisited.pop(0)]

            changed = True
            while changed:
                changed = False
                still_outside = []

                for candidate in unvisited:
                    merged = False

                    for member in cluster_entries:
                        dist = math.hypot(
                            candidate["x"] - member["x"],
                            candidate["y"] - member["y"],
                        )

                        if dist <= self.cluster_pickup_dist_in:
                            cluster_entries.append(candidate)
                            merged = True
                            changed = True
                            break

                    if not merged:
                        still_outside.append(candidate)

                unvisited = still_outside

            total_score = sum(entry["score"] for entry in cluster_entries)
            total_weight = max(total_score, 1e-6)

            cx = (
                sum(entry["x"] * entry["score"] for entry in cluster_entries)
                / total_weight
            )
            cy = (
                sum(entry["y"] * entry["score"] for entry in cluster_entries)
                / total_weight
            )

            clusters.append(
                {
                    "x": cx,
                    "y": cy,
                    "score": total_score,
                    "source_indices": [entry["source_index"] for entry in cluster_entries],
                    "class_name": cluster_entries[0]["class_name"],
                    "members": cluster_entries,
                }
            )

        return clusters

    # -------------------------------------------------------------------------
    # Scoring/filtering helpers
    # -------------------------------------------------------------------------

    def detect_color_from_class(self, class_name: str) -> Optional[str]:
        name = class_name.lower()

        if "red" in name:
            return "red"

        if "blue" in name:
            return "blue"

        return None

    def cluster_counts(
        self,
        x: float,
        y: float,
        detections: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        same = 0
        opp = 0

        for det in detections:
            if not isinstance(det, dict):
                continue

            class_name = str(det.get("class_name", det.get("class", "")))
            color = self.detect_color_from_class(class_name)

            if color not in ("red", "blue"):
                continue

            x_field = det.get("x_field_in")
            y_field = det.get("y_field_in")

            if x_field is None or y_field is None:
                continue

            ox = self.safe_float(x_field)
            oy = self.safe_float(y_field)

            if not (math.isfinite(ox) and math.isfinite(oy)):
                continue

            dist = math.hypot(ox - x, oy - y)

            if dist <= 1e-6:
                continue

            if dist < self.cluster_dist_in:
                if color == self.target_color:
                    same += 1
                elif color == self.opponent_color:
                    opp += 1

        return same, opp

    def score_candidate(
        self,
        x: float,
        y: float,
        same_nearby: int,
        opp_nearby: int,
    ) -> float:
        if same_nearby >= 3:
            base = 12.0 + same_nearby
        elif same_nearby >= 1:
            base = 8.0 + same_nearby
        else:
            if opp_nearby == 0:
                base = 5.0
            elif opp_nearby == 1:
                base = 3.5
            else:
                base = 0.5

        if self.near_wall_count(x, y) == 1:
            base *= 0.6

        return base

    def near_wall_count(self, x: float, y: float) -> int:
        count = 0

        if x < self.field_min_x_in + self.wall_margin_in:
            count += 1

        if x > self.field_max_x_in - self.wall_margin_in:
            count += 1

        if y < self.field_min_y_in + self.wall_margin_in:
            count += 1

        if y > self.field_max_y_in - self.wall_margin_in:
            count += 1

        return count

    def near_any_goal(self, x: float, y: float) -> bool:
        field_cx = 0.5 * (self.field_min_x_in + self.field_max_x_in)
        field_cy = 0.5 * (self.field_min_y_in + self.field_max_y_in)

        pad = self.goal_skip_pad_in

        # Center X structure: two bars crossing at the field center, each rotated
        # 45deg, center_x_thickness_in (5in) thick and reaching center_x_radius_in
        # (11in) out along its diagonal -> a 22in-diameter X. A block is on the X
        # if it lies within half-thickness of one diagonal AND within the arm
        # length along it. p/q are distances to the two diagonals (y=x, y=-x).
        dx = x - field_cx
        dy = y - field_cy
        half_t = 0.5 * self.center_x_thickness_in + pad
        half_len = self.center_x_radius_in + pad
        p = abs(dx - dy) / math.sqrt(2.0)   # to the y=x diagonal
        q = abs(dx + dy) / math.sqrt(2.0)   # to the y=-x diagonal
        if (p <= half_t and q <= half_len) or (q <= half_t and p <= half_len):
            return True

        # Left long goal: explicit rectangle (measured corners), grown by pad.
        if (
            self.long_goal_x_min_in - pad <= x <= self.long_goal_x_max_in + pad
            and self.long_goal_y_min_in - pad <= y <= self.long_goal_y_max_in + pad
        ):
            return True

        # Right long goal: mirror of the left across the field center-x.
        right_x_min = self.field_max_x_in - self.long_goal_x_max_in
        right_x_max = self.field_max_x_in - self.long_goal_x_min_in
        if (
            right_x_min - pad <= x <= right_x_max + pad
            and self.long_goal_y_min_in - pad <= y <= self.long_goal_y_max_in + pad
        ):
            return True

        # Bottom horizontal goal zone
        if (
            y < self.field_min_y_in + self.top_bottom_goal_y_margin_in
            and self.top_bottom_goal_x_min_in <= x <= self.top_bottom_goal_x_max_in
        ):
            return True

        # Top horizontal goal zone
        if (
            y > self.field_max_y_in - self.top_bottom_goal_y_margin_in
            and self.top_bottom_goal_x_min_in <= x <= self.top_bottom_goal_x_max_in
        ):
            return True

        return False

    def in_field_bounds(self, x: float, y: float) -> bool:
        return (
            self.field_min_x_in <= x <= self.field_max_x_in
            and self.field_min_y_in <= y <= self.field_max_y_in
        )

    def heading_from_to(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
    ) -> float:
        dx = to_x - from_x
        dy = to_y - from_y

        # Heading convention: 0 deg = +Y, signed shortest rotation in (-180, 180].
        # +ve = turn toward +X (right), -ve = turn toward -X (left).
        # atan2 already returns (-180, 180], so no wrap-to-360 is applied.
        return math.degrees(math.atan2(dx, dy))

    @staticmethod
    def safe_float(value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # -------------------------------------------------------------------------
    # Publishers
    # -------------------------------------------------------------------------

    def publish_route(
        self,
        route: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        reason: str,
    ):
        self.route_sequence += 1

        points = [
            {
                "x": point["x"],
                "y": point["y"],
                "heading": point["heading"],
            }
            for point in route
        ]

        packet = {
            "type": "ROUTE_SUGGESTION",
            "id": self.route_sequence,
            "stamp_ns": self.get_clock().now().nanoseconds,
            "reason": reason,
            "alliance_color": self.alliance_color,
            "target_color": self.target_color,
            "opponent_color": self.opponent_color,
            "robot_pose_valid": self.robot_pose_valid,
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_heading_deg": self.robot_heading_deg,
            "action_id": ACTION_ID_COLLECT_BLOCKS,
            "action_name": ACTION_NAME_COLLECT_BLOCKS,
            "points": points,
            "route_debug": route,
            "candidate_count": len(candidates),
            "params": {
                "source": "route_planner",
                "alliance_color": self.alliance_color,
                "target_color": self.target_color,
                "intake_speed": 100,
            },
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.route_pub.publish(msg)

    def publish_status(
        self,
        route: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        reason: str,
        detection_age: Optional[float],
        odom_age: Optional[float],
    ):
        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "status": reason,
            "alliance_color": self.alliance_color,
            "target_color": self.target_color,
            "opponent_color": self.opponent_color,
            "robot_pose_valid": self.robot_pose_valid,
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_heading_deg": self.robot_heading_deg,
            "detection_age_sec": detection_age,
            "odom_age_sec": odom_age,
            "candidate_count": len(candidates),
            "route_point_count": len(route),
            "route": route,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = RoutePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Route planner stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()