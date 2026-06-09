import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Leo's on-robot test of the GENERAL (scripted) strategy for the 24in robot.
#
# What this runs (full closed loop, REAL robot pose):
#   serial_messenger  -- talks to the V5/ESP32 over USB. Publishes the robot's
#                        REAL coordinates on /v5/odom (field inches) and forwards
#                        approved /robot/command actions to the V5. Zeros the gyro
#                        at startup (square the robot up at heading 0 first).
#   perception        -- OAK camera + YOLO -> /perception/oak_front/detections,
#                        projected to field inches using the live /v5/odom pose.
#   route_planner     -- turns detected blocks into a COLLECT route, kept to the
#                        LEFT half of the field for the 24in robot.
#   decision          -- safety gate; forwards approved actions to the V5.
#   policy (scripted) -- the general game plan: scan -> assess -> collect (via the
#                        route planner) / score / descore -> repeat. Waypoints are
#                        clamped to the LEFT half.
#
# Unlike integration_policy_test, there is NO injected/fake odom and NO test
# driver -- the robot's own coordinates from the serial messenger drive the run.
#
# Units are INCHES end to end. Heading is degrees, signed (-180, 180], 0 = +Y.
# Field is 144 in x 144 in, origin bottom-left.
#
# 24in robot stays on the LEFT half (x in [0, 72]) via two independent limits:
#   - route planner pickup zone: pickup_x_min_in / pickup_x_max_in
#   - scripted strategy waypoint clamp: zone_x_min_in / zone_x_max_in
#
# Run:
#   ros2 launch leroi_bringup leo_scripted_24in_test.launch.py
#   # then confirm each proposed action (safe default):
#   ros2 topic pub --once /policy/decision std_msgs/String "{data: 'confirm'}"
#   # or run hands-off:
#   ros2 launch leroi_bringup leo_scripted_24in_test.launch.py require_confirmation:=false


YOLO_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/yolo/best_fp16.engine"
)

FIELD_MAP_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/field_map.png"
)

POLICY_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/policy/policy.onnx"
)

# Half-field limits for the 24in robot (LEFT half), field inches.
LEFT_X_MIN_IN = 0.0
LEFT_X_MAX_IN = 72.0   # field center


def generate_launch_description():
    # --- Serial bridge to the V5: REAL robot coordinates on /v5/odom ---
    serial_messenger = Node(
        package="leroi_serial_messenger",
        executable="serial_messenger_node",
        name="serial_messenger_node",
        output="screen",
        parameters=[
            {
                # Leave empty to auto-detect the VEX V5 user port.
                "serial_port": "",
                "baud_rate": 115200,
                "read_timeout_sec": 0.05,
                "reconnect_period_sec": 2.0,

                # Serial TX
                "tx_append_newline": True,
                "max_payload_bytes": 1024,
                "drop_tx_when_disconnected": True,

                # Heartbeat/status
                "send_heartbeat": True,
                "heartbeat_period_sec": 5.0,
                "status_period_sec": 2.0,

                # Debugging (on for a supervised test)
                "debug_log_rx": True,
                "debug_log_tx": True,
                "debug_log_scan": True,

                # Not forwarding raw detections to the V5 in this test.
                "detection_topic": "/perception/oak_front/detections",
                "forward_detections_to_v5": False,
                "debug_log_detections": False,

                # Gyro/odom-board reset. Square the robot up at heading 0 by hand,
                # then it zeros once at startup and again on any RESET_GYRO ack.
                "gyro_reset_on_ack": True,
                "gyro_reset_on_startup": True,
                "gyro_startup_reset_delay_sec": 2.0,
                "gyro_reset_char": "R",
            }
        ],
    )

    # --- Perception: OAK + YOLO -> field-projected detections (inches) ---
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
                # Color-bearing class names so the route planner's color filter
                # works (otherwise detections are class0/class2 -> NO_ROUTE).
                "yolo_class_names": '{"0": "blue_block", "1": "blue_block_in_goal", "2": "red_block", "3": "red_block_in_goal"}',
                "yolo_confidence": 0.35,
                "yolo_image_size": 320,
                "yolo_device": "0",
                "yolo_use_half": False,

                "enable_preview_stream": True,
                "preview_port": 8080,
                "preview_jpeg_quality": 60,
                "draw_detections": True,
                "enable_map_stream": True,
                "map_width_px": 700,
                "map_height_px": 700,

                # Field bounds (inches)
                "field_min_x_in": 0.0,
                "field_max_x_in": 144.0,
                "field_min_y_in": 0.0,
                "field_max_y_in": 144.0,
                "field_map_image_path": FIELD_MAP_PATH,
                "field_map_rotation": "cw",
                "field_map_alpha": 0.72,
                "draw_field_grid": True,

                "robot_trail_max_points": 500,
                "object_history_sec": 3.0,

                "target_path_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",

                "camera_offset_x_in": -0.25,  # camera mounted 0.25in left of robot center
                "camera_offset_y_in": 5.91,
                "camera_offset_z_in": 9.84,

                # Before the V5 streams odom, project from a LEFT-side start pose so
                # early detections land in-bounds. Overwritten by real /v5/odom.
                "assume_origin_when_no_odom": True,
                "assumed_start_x_in": 24.0,
                "assumed_start_y_in": 12.0,
                "assumed_start_heading_deg": 0.0,

                "enable_goal_view": False,
            }
        ],
    )

    # --- Route planner: COLLECT routes, LEFT half only for the 24in robot ---
    route_planner = Node(
        package="leroi_perception",
        executable="route_planner_node",
        name="route_planner_node",
        output="screen",
        parameters=[
            {
                "alliance_color": LaunchConfiguration("alliance_color"),
                "detections_topic": "/perception/oak_front/detections",
                "odom_topic": "/v5/odom",
                "route_suggestion_topic": "/perception/route_suggestion",
                "route_status_topic": "/perception/route_status",
                "route_rate_hz": 3.0,
                "stale_detection_timeout_sec": 1.0,
                "stale_odom_timeout_sec": 1.0,
                # Require fresh real odom from the V5 before planning a route.
                "require_fresh_odom": True,

                "field_min_x_in": 0.0,
                "field_max_x_in": 144.0,
                "field_min_y_in": 0.0,
                "field_max_y_in": 144.0,

                # LEFT-half pickup zone for the 24in robot.
                "pickup_x_min_in": LEFT_X_MIN_IN,
                "pickup_x_max_in": LEFT_X_MAX_IN,

                "max_route_points": 5,
                "max_carry_blocks": 5,
                "already_held_blocks": 0,
                "min_confidence": 0.35,
                "wall_margin_in": 17.72,
                "cluster_dist_in": 35.43,
                "cluster_pickup_dist_in": 14.96,
                "min_route_value": 0.0381,

                # Field structures to avoid picking up inside (measured geometry).
                "skip_near_goals": True,
                # Center X: 22in diameter (radius 11), 5in-thick diagonal bars.
                "center_x_radius_in": 11.0,
                "center_x_thickness_in": 5.0,
                # Long goal rectangle (corners 21,48 - 27,96); right side mirrored.
                "long_goal_x_min_in": 21.0,
                "long_goal_x_max_in": 27.0,
                "long_goal_y_min_in": 48.0,
                "long_goal_y_max_in": 96.0,
                "goal_skip_pad_in": 0.0,
                # Top/bottom (short) goals -- placeholder until measured.
                "top_bottom_goal_x_min_in": 47.24,
                "top_bottom_goal_x_max_in": 96.85,
                "top_bottom_goal_y_margin_in": 21.65,

                "publish_empty_route": True,
                "debug_log_routes": True,
            }
        ],
    )

    # --- Decision: safety gate, forwards approved actions to /robot/command ---
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
                "allow_score_without_blocks": False,
                "block_new_action_while_waiting": True,
                "allow_stop_while_waiting": True,
                # RESET_GYRO (24) may interrupt an in-flight action.
                "allow_gyro_reset_while_waiting": True,
            }
        ],
    )

    # --- Policy: GENERAL (scripted) strategy, 24in robot, LEFT half ---
    policy = Node(
        package="leroi_policy",
        executable="policy_node",
        name="policy_node",
        output="screen",
        parameters=[
            {
                # Inputs (game state)
                "mapped_objects_topic": "/perception/oak_front/mapped_objects",
                "detections_topic": "/perception/oak_front/detections",
                "route_suggestion_topic": "/perception/route_suggestion",
                "odom_topic": "/v5/odom",
                "decision_status_topic": "/decision/status",

                # Output + operator gate
                "action_request_topic": "/decision/action_request",
                "pending_action_topic": "/policy/pending_action",
                "decision_topic": "/policy/decision",
                "status_topic": "/policy/status",

                # General strategy = scripted game plan.
                "strategy_mode": LaunchConfiguration("strategy_mode"),

                # RL fields (unused when scripted, declared for completeness).
                "model_path": POLICY_MODEL_PATH,
                "onnx_providers": "CPUExecutionProvider",
                "obs_dim_override": 0,
                "action_id_map": "",

                # Scripted strategy. 0 ids -> alliance-aware default.
                "alliance_color": LaunchConfiguration("alliance_color"),
                "score_long_action_id": 0,
                "score_midlow_action_id": 0,
                "descore_action_id": 0,
                "max_collect_targets": 3,
                # TODO: set the real LEFT-side scan/fallback field coords (inches).
                "scan_point_blue_json": "",
                "scan_point_red_json": "",
                "collect_fallback_blue_json": "",
                "collect_fallback_red_json": "",
                "goal_regions_json": "",
                "goal_color_margin": 1,
                "goal_empty_max": 1,
                "goal_full_min": 4,

                # Collect via the route planner; clamp waypoints to the LEFT half.
                "use_route_planner_for_collect": True,
                "zone_x_min_in": LEFT_X_MIN_IN,
                "zone_x_max_in": LEFT_X_MAX_IN,
                "zone_x_wander_in": 12.0,

                # Behavior
                "inference_rate_hz": 1.0,
                "require_confirmation": LaunchConfiguration("require_confirmation"),
                "respect_decision_busy": True,
                "max_blocks": 10,
                "field_size_in": 144.0,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "alliance_color",
                default_value="blue",
                description="Alliance for the scripted strategy: blue or red",
            ),
            DeclareLaunchArgument(
                "strategy_mode",
                default_value="scripted",
                description="Policy strategy: 'scripted' (general plan) or 'rl'",
            ),
            DeclareLaunchArgument(
                "require_confirmation",
                default_value="true",
                description="Hold each proposed action until 'confirm' on "
                "/policy/decision. Set false to let the 24in robot run hands-off.",
            ),
            serial_messenger,
            perception,
            route_planner,
            decision,
            policy,
        ]
    )
