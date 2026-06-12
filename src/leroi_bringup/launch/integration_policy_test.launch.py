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
                "debug_log_tx": True,  # print each command sent to the V5
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
                # Map YOLO class ids -> color-bearing names so the route planner's
                # color filter (looks for "blue"/"red" in class_name) works. Without
                # this, detections come out as "class0"/"class2" and no candidates
                # are ever produced -> NO_ROUTE.
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
                # Always score the LEFT long goal (action 3,
                # SCORE_LEFT_LONG_GOAL_ALLIANCE). score_long stays 3 (blue
                # default); the mid/low slot is pointed at 3 too so the
                # assess branch that used to fire SCORE_MID_LEFT now scores long.
                "score_long_action_id": 3,
                "score_midlow_action_id": 3,
                "descore_action_id": 0,
                # Drop the MOVE_CHASSIS_ONLY scan step: assess from wherever the
                # robot is and loop ASSESS -> COLLECT -> SCORE -> ASSESS.
                "scan_before_assess": False,
                # Forced demo loop: always COLLECT_BLOCKS -> SCORE_LONG (no goal
                # assessment / descore), and stop after 2 score-long cycles.
                "force_collect_score_long": True,
                "max_loops": 2,
                # Never score the mid goal: if the plan would, drive home to
                # (x=48, y=28) instead and skip the mid-goal score.
                "avoid_mid_goal": True,
                "mid_goal_home_json": '[{"x": 48.0, "y": 28.0, "heading": 0.0}]',
                "max_collect_targets": 3,
                # Alliance-specific positions (inches). Scan point = near the
                # parking zone, measured on the 24in robot.
                # TODO: set the red scan point (and confirm blue alliance).
                "scan_point_blue_json": '[{"x": 43.0, "y": 27.0, "heading": -18.0}]',
                "scan_point_red_json": "",
                "collect_fallback_blue_json": "",
                "collect_fallback_red_json": "",
                # Goal-state assessment (regions in field in). TODO: set real goals.
                "goal_regions_json": "",
                "goal_color_margin": 1,
                "goal_empty_max": 1,
                "goal_full_min": 4,
                # Collect routing + half-of-field zone. Defaults keep the 24in
                # robot on the LEFT half (x in [0, 72]); for the 15in robot run
                # with zone_x_min_in:=72 zone_x_max_in:=144 (RIGHT half).
                "use_route_planner_for_collect": True,
                "zone_x_min_in": LaunchConfiguration("zone_x_min_in"),
                "zone_x_max_in": LaunchConfiguration("zone_x_max_in"),
                "zone_x_wander_in": 12.0,

                # Behavior
                "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
                "require_confirmation": LaunchConfiguration("require_confirmation"),
                "respect_decision_busy": True,
                # Wait this long after the robot goes idle before the next action.
                "decision_delay_sec": LaunchConfiguration("decision_delay_sec"),
                "max_blocks": 10,
                "field_size_in": 144.0,
            }
        ],
    )

    # Route planner: produces COLLECT_BLOCKS routes the scripted strategy consumes.
    # pickup_x_max_in keeps the 24in robot on the LEFT half (72in)
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
                "field_min_x_in": 0.0,
                "field_max_x_in": 144.0,
                "field_min_y_in": 0.0,
                "field_max_y_in": 144.0,
                # Half-of-field pickup zone. Defaults = LEFT half for the 24in
                # robot ([0, 72]); flip with pickup_x_min_in:=72 pickup_x_max_in:=144
                # for the 15in robot (RIGHT half).
                "pickup_x_min_in": LaunchConfiguration("pickup_x_min_in"),
                "pickup_x_max_in": LaunchConfiguration("pickup_x_max_in"),
                "max_route_points": 5,
                "max_carry_blocks": 5,
                "already_held_blocks": 0,
                "min_confidence": 0.35,
                "wall_margin_in": 17.72,
                "cluster_dist_in": 35.43,
                "cluster_pickup_dist_in": 14.96,
                "min_route_value": 0.0381,
                "skip_near_goals": True,
                # Center X: 22in diameter (radius 11), 5in-thick diagonal bars.
                "center_x_radius_in": 11.0,
                "center_x_thickness_in": 5.0,
                # Long goal = measured rectangle (corners 21,48 - 27,96); right
                # side mirrored automatically across field center-x -> [117,123].
                "long_goal_x_min_in": 21.0,
                "long_goal_x_max_in": 27.0,
                "long_goal_y_min_in": 48.0,
                "long_goal_y_max_in": 96.0,
                "goal_skip_pad_in": 0.0,
                "top_bottom_goal_x_min_in": 47.24,
                "top_bottom_goal_x_max_in": 96.85,
                "top_bottom_goal_y_margin_in": 21.65,
                "publish_empty_route": True,
                "debug_log_routes": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "strategy_mode",
                default_value="scripted",
                description="Policy strategy: 'rl' (ONNX model) or 'scripted' (defined plan)",
            ),
            DeclareLaunchArgument(
                "alliance_color",
                default_value="blue",
                description="Alliance for the scripted strategy: blue or red",
            ),
            DeclareLaunchArgument(
                "zone_x_min_in",
                default_value="0.0",
                description="Soft min-x (in) for waypoints. 0 for the LEFT (24in) robot; 72 for the RIGHT (15in) robot",
            ),
            DeclareLaunchArgument(
                "zone_x_max_in",
                default_value="72.0",
                description="Soft max-x (in) for waypoints. 72 for the LEFT (24in) robot; 144 for the RIGHT (15in) robot",
            ),
            DeclareLaunchArgument(
                "pickup_x_min_in",
                default_value="0.0",
                description="Route planner min block-pickup x (in). 0 for LEFT (24in); 72 for RIGHT (15in)",
            ),
            DeclareLaunchArgument(
                "pickup_x_max_in",
                default_value="72.0",  # 72 in = field center
                description="Route planner max block-pickup x (in). 72 for LEFT (24in); 144 for RIGHT (15in)",
            ),
            DeclareLaunchArgument(
                "policy_model_path",
                default_value=POLICY_MODEL_PATH,
                description="Path to the exported ONNX policy model",
            ),
            DeclareLaunchArgument(
                "require_confirmation",
                default_value="false",
                description="Hold each proposed action until confirmed on /policy/decision. Default false -> run autonomously.",
            ),
            DeclareLaunchArgument(
                "inference_rate_hz",
                default_value="4.0",
                description="How often the policy proposes an action when idle (Hz). 4 Hz keeps the inter-decision delay tight.",
            ),
            DeclareLaunchArgument(
                "decision_delay_sec",
                default_value="1.5",
                description="Seconds to wait after the robot goes idle before sending the next action (1-2s pause between decisions)",
            ),
            serial_messenger,
            perception,
            route_planner,
            decision,
            policy,
        ]
    )
