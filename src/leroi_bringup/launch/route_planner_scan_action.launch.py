"""Route planner + scan + score/descore test (follow-up to route_planner_test).

Flow (one action per ACK, each held behind a confirm/cancel gate):

  perception (camera + YOLO) -> route planner (left-half pickup zone)
    -> policy_node in 'scan_score' mode:
         COLLECT (route planner pickup) -> SCAN (drive to scan point, face goal)
         -> ASSESS long goal:
              empty / partial (not full)  -> SCORE long goal
              majority opponent blocks    -> DESCORE
         -> loop
    -> decision_node -> serial_messenger_node -> V5

See what it wants to do, then confirm/deny:

    ros2 topic echo /policy/pending_action     # proposed action + assessment (info)
    ros2 topic echo /policy/status             # phase + last_assessment
    ros2 topic pub --once /policy/decision std_msgs/String "data: 'confirm'"
    ros2 topic pub --once /policy/decision std_msgs/String "data: 'cancel'"

Vision feed: perception serves an MJPEG preview (camera + detections + field map) on
http://<jetson>:8080 — SSH-forward port 8080 to see what the robot sees.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


YOLO_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/yolo/best_fp16.engine"
)
FIELD_MAP_PATH = os.path.expanduser("~/leroiai-jetsonWorkspace/field_map.png")

# Blue scan point measured on the 24in robot (inches): near parking zone, facing goal.
SCAN_POINT_BLUE = '[{"x": 43.0, "y": 27.0, "heading": -18.0}]'


def generate_launch_description():
    alliance_color = LaunchConfiguration("alliance_color")

    serial_messenger = Node(
        package="leroi_serial_messenger",
        executable="serial_messenger_node",
        name="serial_messenger_node",
        output="screen",
        parameters=[
            {
                "serial_port": "",
                "baud_rate": 115200,
                "read_timeout_sec": 0.05,
                "reconnect_period_sec": 2.0,
                "tx_append_newline": True,
                "max_payload_bytes": 1024,
                "drop_tx_when_disconnected": True,
                "send_heartbeat": True,
                "heartbeat_period_sec": 5.0,
                "status_period_sec": 2.0,
                "debug_log_rx": False,
                "debug_log_tx": True,  # print each command sent to the V5
                "debug_log_scan": False,  # quiet the port-scan spam
            }
        ],
    )

    perception = Node(
        package="leroi_perception",
        executable="perception_node",
        name="oak_front_perception_node",
        output="screen",
        parameters=[
            {
                "camera_name": "oak_front",
                "camera_type": "oak",
                "camera_index": 0,
                "camera_width": 640,
                "camera_height": 480,
                "camera_fps": 30,
                "oak_mxid": "",
                "oak_with_depth": True,
                "preview_rate_hz": 30.0,
                "preview_stream_fps": 15.0,
                "status_rate_hz": 2.0,
                "enable_yolo": True,
                "yolo_rate_hz": 15.0,
                "yolo_model_path": YOLO_MODEL_PATH,
                "yolo_confidence": 0.35,
                "yolo_image_size": 320,
                "yolo_device": "0",
                "yolo_use_half": False,
                # Vision feed: MJPEG preview (camera + detections) on :8080.
                "enable_preview_stream": True,
                "preview_port": 8080,
                "preview_jpeg_quality": 60,
                "draw_detections": True,
                "enable_map_stream": True,
                "map_width_px": 700,
                "map_height_px": 700,
                "field_min_x_cm": 0.0,
                "field_max_x_cm": 144.0,  # inches now
                "field_min_y_cm": 0.0,
                "field_max_y_cm": 144.0,  # inches now
                "field_map_image_path": FIELD_MAP_PATH,
                "field_map_rotation": "cw",
                "field_map_alpha": 0.72,
                "draw_field_grid": True,
                "robot_trail_max_points": 500,
                "object_history_sec": 3.0,
                "target_path_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",
                # Camera mount on the 24in robot. TODO: re-measure/rescale if the
                # projected distances look off (see camera_offset_* and depth).
                "camera_offset_x_cm": -0.635,
                "camera_offset_y_cm": 15.0,
                "camera_offset_z_cm": 25.0,
                "enable_goal_view": False,
            }
        ],
    )

    route_planner = Node(
        package="leroi_perception",
        executable="route_planner_node",
        name="route_planner_node",
        output="screen",
        parameters=[
            {
                "alliance_color": alliance_color,
                # YOLO labels are class0/class1/...; map them to colors.
                "class_color_map": '{"class0": "blue", "class1": "red"}',
                "detections_topic": "/perception/oak_front/detections",
                "odom_topic": "/v5/odom",
                "route_suggestion_topic": "/perception/route_suggestion",
                "route_status_topic": "/perception/route_status",
                "route_rate_hz": 3.0,
                "stale_detection_timeout_sec": 1.0,
                "stale_odom_timeout_sec": 1.0,
                "field_min_x_cm": 0.0,
                "field_max_x_cm": 144.0,  # inches now
                "field_min_y_cm": 0.0,
                "field_max_y_cm": 144.0,  # inches now
                "pickup_x_min_cm": 0.0,
                "pickup_x_max_cm": LaunchConfiguration("pickup_x_max_cm"),
                "max_route_points": 5,
                "max_carry_blocks": 5,
                "already_held_blocks": 0,
                "min_confidence": 0.35,
                "wall_margin_cm": 45.0,
                "cluster_dist_cm": 90.0,
                "cluster_pickup_dist_cm": 38.0,
                "min_route_value": 0.015,
                "skip_near_goals": True,
                "center_goal_skip_radius_cm": 65.0,
                "long_goal_y_min_cm": 115.0,
                "long_goal_y_max_cm": 250.0,
                "long_goal_x_margin_cm": 75.0,
                "top_bottom_goal_x_min_cm": 120.0,
                "top_bottom_goal_x_max_cm": 246.0,
                "top_bottom_goal_y_margin_cm": 55.0,
                "publish_empty_route": True,
                "debug_log_routes": False,  # quieter terminal
            }
        ],
    )

    decision = Node(
        package="leroi_decision",
        executable="decision_node",
        name="decision_node",
        output="screen",
        parameters=[
            {
                "action_request_topic": "/decision/action_request",
                "robot_command_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",
                "status_topic": "/decision/status",
                "rejected_topic": "/decision/rejected",
                "command_log_topic": "/decision/command_log",
                "command_log_path": os.path.expanduser(
                    "~/leroiai-jetsonWorkspace/logs/decision_commands.jsonl"
                ),
                "command_history_max": 100,
                "status_rate_hz": 2.0,
                "command_timeout_sec": 15.0,
                # Allow scoring in this test even though we don't track the hopper.
                "allow_score_without_blocks": True,
                "block_new_action_while_waiting": True,
                "allow_stop_while_waiting": True,
                "reject_unknown_action_ids": True,
            }
        ],
    )

    policy = Node(
        package="leroi_policy",
        executable="policy_node",
        name="policy_node",
        output="screen",
        parameters=[
            {
                "mapped_objects_topic": "/perception/oak_front/mapped_objects",
                "detections_topic": "/perception/oak_front/detections",
                "route_suggestion_topic": "/perception/route_suggestion",
                "odom_topic": "/v5/odom",
                "decision_status_topic": "/decision/status",
                "action_request_topic": "/decision/action_request",
                "pending_action_topic": "/policy/pending_action",
                "decision_topic": "/policy/decision",
                "status_topic": "/policy/status",
                # collect -> scan -> decide (descore/score) -> loop.
                "strategy_mode": "scan_score",
                "alliance_color": alliance_color,
                "score_long_action_id": 0,   # 0 -> alliance default (blue 3 / red 5)
                "descore_action_id": 0,      # 0 -> alliance default (blue 14 / red 18)
                "max_collect_targets": 5,
                "scan_point_blue_json": SCAN_POINT_BLUE,
                "scan_point_red_json": "",   # TODO: red scan point
                # Long-goal decision uses the YOLO class (in-goal), not x/y.
                # Default: always descore (harmless) then score. Set false to use
                # the class assessment (majority opponent -> descore, else score).
                "fallback_descore_then_score": LaunchConfiguration("fallback_descore_then_score"),
                "in_goal_keywords": "goal,in_goal,scored",
                "long_goal_capacity": 14,
                "use_route_planner_for_collect": True,
                "zone_x_max_in": 72.0,
                "zone_x_wander_in": 12.0,
                "inference_rate_hz": 2.0,
                "require_confirmation": LaunchConfiguration("require_confirmation"),
                "respect_decision_busy": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "alliance_color",
                default_value="blue",
                description="Alliance (route target color + goal selection): blue or red",
            ),
            DeclareLaunchArgument(
                "require_confirmation",
                default_value="true",
                description="Hold each proposed action until confirmed on /policy/decision",
            ),
            DeclareLaunchArgument(
                "fallback_descore_then_score",
                default_value="true",
                description="true: always descore then score; false: class-based assess",
            ),
            DeclareLaunchArgument(
                "pickup_x_max_cm",
                default_value="72.0",  # inches: keep the 24in robot on the left half
                description="Route planner: max block-pickup x (cm)",
            ),
            serial_messenger,
            perception,
            route_planner,
            decision,
            policy,
        ]
    )
