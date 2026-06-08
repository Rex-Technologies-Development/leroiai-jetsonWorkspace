import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


YOLO_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/yolo/best_fp16.engine"
)

FIELD_MAP_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/field_map.png"
)

POLICY_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/policy/policy.onnx"
)


def generate_launch_description():
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
                "debug_log_tx": False,
                "debug_log_scan": True,
                "detection_topic": "/perception/oak_front/detections",
                "forward_detections_to_v5": False,
                "debug_log_detections": False,
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
                "allow_score_without_blocks": False,
                "block_new_action_while_waiting": True,
                "allow_stop_while_waiting": True,
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
                "enable_preview_stream": True,
                "preview_port": 8080,
                "preview_jpeg_quality": 60,
                "draw_detections": True,
                "enable_map_stream": True,
                "map_width_px": 700,
                "map_height_px": 700,
                "field_min_x_cm": 0.0,
                "field_max_x_cm": 365.76,
                "field_min_y_cm": 0.0,
                "field_max_y_cm": 365.76,
                "field_map_image_path": FIELD_MAP_PATH,
                "field_map_rotation": "cw",
                "field_map_alpha": 0.72,
                "draw_field_grid": True,
                "robot_trail_max_points": 500,
                "object_history_sec": 3.0,
                "target_path_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",
                "camera_offset_x_cm": -0.635,  # camera mounted 0.25in left of robot center
                "camera_offset_y_cm": 15.0,
                "camera_offset_z_cm": 25.0,
                "enable_goal_view": False,
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

                # Strategy: "rl" (ONNX model) or "scripted" (defined game plan).
                "strategy_mode": LaunchConfiguration("strategy_mode"),

                # RL strategy
                "model_path": LaunchConfiguration("policy_model_path"),
                # CPU is plenty for a small policy MLP and keeps the GPU free for
                # perception's YOLO TensorRT engine. To use the GPU, install the
                # Jetson onnxruntime-gpu build and set e.g.:
                # "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
                "onnx_providers": "CPUExecutionProvider",
                "obs_dim_override": 0,
                "action_id_map": "",

                # Scripted strategy (defined game plan). 0 ids -> alliance default.
                # Plan: scan near parking zone -> assess goals -> descore /
                # collect+score-midlow / collect+score-long -> back to scan.
                "alliance_color": LaunchConfiguration("alliance_color"),
                "score_long_action_id": 0,
                "score_midlow_action_id": 0,
                "descore_action_id": 0,
                "max_collect_targets": 3,
                # Alliance-specific positions (inches). TODO: set real field coords.
                "scan_point_blue_json": "",
                "scan_point_red_json": "",
                "collect_fallback_blue_json": "",
                "collect_fallback_red_json": "",
                # Goal-state assessment (regions in field cm). TODO: set real goals.
                "goal_regions_json": "",
                "goal_color_margin": 1,
                "goal_empty_max": 1,
                "goal_full_min": 4,
                # Collect routing + left-half zone for the 24in robot.
                "use_route_planner_for_collect": True,
                "zone_x_max_in": LaunchConfiguration("zone_x_max_in"),
                "zone_x_wander_in": 12.0,

                # Behavior
                "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
                "require_confirmation": LaunchConfiguration("require_confirmation"),
                "respect_decision_busy": True,
                "max_blocks": 10,
                "field_size_cm": 365.76,
            }
        ],
    )

    # Route planner: produces COLLECT_BLOCKS routes the scripted strategy consumes.
    # pickup_x_max_cm keeps the 24in robot on the LEFT half (72in -> 182.88cm)
    # so it does not impede the 15in robot on the right.
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
                "field_min_x_cm": 0.0,
                "field_max_x_cm": 365.76,
                "field_min_y_cm": 0.0,
                "field_max_y_cm": 365.76,
                # Left-half pickup zone for the 24in robot (cm).
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
                "debug_log_routes": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "strategy_mode",
                default_value="rl",
                description="Policy strategy: 'rl' (ONNX model) or 'scripted' (defined plan)",
            ),
            DeclareLaunchArgument(
                "alliance_color",
                default_value="blue",
                description="Alliance for the scripted strategy: blue or red",
            ),
            DeclareLaunchArgument(
                "zone_x_max_in",
                default_value="72.0",
                description="Soft left-half limit (inches) for the 24in robot's waypoints",
            ),
            DeclareLaunchArgument(
                "pickup_x_max_cm",
                default_value="182.88",  # 72 in
                description="Route planner: max block-pickup x (cm); keeps 24in robot left",
            ),
            DeclareLaunchArgument(
                "policy_model_path",
                default_value=POLICY_MODEL_PATH,
                description="Path to the exported ONNX policy model",
            ),
            DeclareLaunchArgument(
                "require_confirmation",
                default_value="true",
                description="Hold each proposed action until confirmed on /policy/decision",
            ),
            DeclareLaunchArgument(
                "inference_rate_hz",
                default_value="1.0",
                description="How often the policy proposes an action when idle",
            ),
            serial_messenger,
            perception,
            route_planner,
            decision,
            policy,
        ]
    )
