import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


class GoalProjector:
    """
    Projects known 3D field coordinates of a high goal into image space
    using robot odometry + camera intrinsics + camera mount transform.

    Coordinate conventions:
      - Field: X = right, Y = forward, Z = up (cm)
      - Robot: same axes, origin at robot center
      - Camera: X = right, Y = down, Z = forward (standard OpenCV)
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        # Goal corners in FIELD coordinates (cm): [top-left, top-right, bottom-right, bottom-left]
        # Order must be consistent so left/right thirds are correct.
        goal_corners_world: List[Tuple[float, float, float]],
        # Camera position relative to robot center (cm): (x_right, y_forward, z_up)
        camera_offset_cm: Tuple[float, float, float] = (0.0, 0.0, 20.0),
        # Camera tilt downward from horizontal (degrees, positive = tilting down)
        camera_tilt_deg: float = 0.0,
    ):
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        self.goal_corners_world = np.array(goal_corners_world, dtype=np.float64)

        # Build camera-to-robot transform (4x4 homogeneous)
        # Camera sits at camera_offset_cm relative to robot center,
        # facing forward (+Y robot) with optional downward tilt.
        ox, oy, oz = camera_offset_cm
        tilt = math.radians(camera_tilt_deg)

        # Rotation: robot +Y maps to camera +Z (forward), robot +X maps to camera +X,
        # robot +Z maps to camera -Y (up in world = up in image when untilted).
        # Then apply downward tilt around camera X axis.
        R_align = np.array([
            [1,  0,           0         ],
            [0,  0,          -1         ],
            [0,  math.cos(tilt), math.sin(tilt)],
        ], dtype=np.float64)

        # Actually use the standard robotics → CV convention:
        # robot frame: X=right, Y=forward, Z=up
        # camera frame: X=right, Y=down, Z=forward
        # Mapping: cam_X = rob_X, cam_Y = -rob_Z + tilt, cam_Z = rob_Y
        ct, st = math.cos(tilt), math.sin(tilt)
        R_robot_to_cam = np.array([
            [1,   0,    0  ],
            [0,   st,  -ct ],
            [0,   ct,   st ],
        ], dtype=np.float64)

        self.T_robot_to_cam = np.eye(4)
        self.T_robot_to_cam[:3, :3] = R_robot_to_cam
        self.T_robot_to_cam[:3,  3] = R_robot_to_cam @ np.array([-ox, -oy, -oz])

    def _world_to_camera(
        self,
        points_world: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
    ) -> np.ndarray:
        """Transform 3D world points into camera frame."""
        theta = math.radians(robot_heading_deg)
        ct, st = math.cos(theta), math.sin(theta)

        # World → Robot transform (inverse of robot pose)
        T_world_to_robot = np.array([
            [ ct,  st, 0, -(ct * robot_x + st * robot_y)],
            [-st,  ct, 0,  (st * robot_x - ct * robot_y)],
            [  0,   0, 1,  0                            ],
            [  0,   0, 0,  1                            ],
        ], dtype=np.float64)

        T_world_to_cam = self.T_robot_to_cam @ T_world_to_robot

        n = len(points_world)
        pts_h = np.hstack([points_world, np.ones((n, 1))])
        pts_cam = (T_world_to_cam @ pts_h.T).T[:, :3]
        return pts_cam

    def project_corners(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Returns projected goal corners as image (u, v) pixel coordinates,
        or None if any corner is behind the camera.
        """
        pts_cam = self._world_to_camera(
            self.goal_corners_world, robot_x, robot_y, robot_heading_deg
        )

        result = []
        for p in pts_cam:
            if p[2] <= 0:
                return None  # corner is behind camera
            u = int(self.K[0, 0] * p[0] / p[2] + self.K[0, 2])
            v = int(self.K[1, 1] * p[1] / p[2] + self.K[1, 2])
            result.append((u, v))
        return result

    def get_third_boundaries(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns (goal_left_x, goal_right_x, third1_x, third2_x) in image pixels.
        third1_x is the LEFT|MIDDLE boundary.
        third2_x is the MIDDLE|RIGHT boundary.
        Returns None if goal is not visible from this pose.
        """
        corners = self.project_corners(robot_x, robot_y, robot_heading_deg)
        if corners is None:
            return None

        xs = [p[0] for p in corners]
        goal_left  = min(xs)
        goal_right = max(xs)
        span = goal_right - goal_left

        if span < 10:  # goal too small to be useful
            return None

        third1_x = goal_left + span // 3
        third2_x = goal_left + 2 * span // 3
        return goal_left, goal_right, third1_x, third2_x

    @staticmethod
    def classify(cx: float, third1_x: int, third2_x: int) -> str:
        if cx < third1_x:
            return "LEFT"
        elif cx < third2_x:
            return "MIDDLE"
        return "RIGHT"

    def draw_overlay(
        self,
        frame: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_heading_deg: float,
        detections=None,
    ) -> np.ndarray:
        """
        Draws projected goal outline, third dividers, and classifies
        any in_goal detections. Returns annotated frame.
        """
        boundaries = self.get_third_boundaries(robot_x, robot_y, robot_heading_deg)
        corners    = self.project_corners(robot_x, robot_y, robot_heading_deg)

        if boundaries is None or corners is None:
            cv2.putText(frame, "GOAL NOT VISIBLE", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        goal_left, goal_right, third1_x, third2_x = boundaries

        # Draw projected goal polygon
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        # Get top/bottom y extent from corners for the divider lines
        ys = [p[1] for p in corners]
        top_y    = min(ys)
        bottom_y = max(ys)

        cv2.line(frame, (third1_x, top_y), (third1_x, bottom_y), (200, 200, 200), 2)
        cv2.line(frame, (third2_x, top_y), (third2_x, bottom_y), (200, 200, 200), 2)

        cv2.putText(frame, "L", (goal_left + 4,  top_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "M", (third1_x  + 4,  top_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "R", (third2_x  + 4,  top_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Classify in-goal detections
        if detections:
            for det in detections:
                if "in_goal" not in str(det.get("class_name", "")):
                    continue
                cx = float(det.get("bbox_center_x",
                           det.get("bbox_x", 0) + det.get("bbox_width", 0) / 2))
                third = self.classify(cx, third1_x, third2_x)
                x = int(det.get("bbox_x", 0))
                y = int(det.get("bbox_y", 0))
                cv2.putText(frame, third, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        return frame


# ---------------------------------------------------------------------------
# VEX AI field geometry helpers
# ---------------------------------------------------------------------------

def make_vex_high_goal_corners(
    goal_center_x_cm: float,
    goal_center_y_cm: float,
    goal_z_bottom_cm: float,
    goal_width_cm: float = 61.0,   # ~24 inches
    goal_height_cm: float = 30.0,  # adjust to match actual goal
) -> List[Tuple[float, float, float]]:
    """
    Returns [top-left, top-right, bottom-right, bottom-left] corners
    of a rectangular high goal face in field coordinates.
    Assumes the goal face is parallel to the X axis (faces along Y).
    """
    hw = goal_width_cm / 2
    zt = goal_z_bottom_cm + goal_height_cm
    zb = goal_z_bottom_cm
    y  = goal_center_y_cm

    return [
        (goal_center_x_cm - hw, y, zt),  # top-left
        (goal_center_x_cm + hw, y, zt),  # top-right
        (goal_center_x_cm + hw, y, zb),  # bottom-right
        (goal_center_x_cm - hw, y, zb),  # bottom-left
    ]
