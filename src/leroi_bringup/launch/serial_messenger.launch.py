from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


YOLO_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/yolo/best_fp16.engine"
)

FIELD_MAP_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/field_map.png"
)


def generate_launch_description():
    alliance_color = LaunchConfiguration("alliance_color")
    field_map_rotation = LaunchConfiguration("field_map_rotation")

    alliance_arg = DeclareLaunchArgument(
        "alliance_color",
        default_value="blue",
        description="Alliance color for route planning: blue or red.",
    )

    field_map_rotation_arg = DeclareLaunchArgument(
        "field_map_rotation",
        default_value="cw",
        description="Field map rotation: none, cw, ccw, or 180.",
    )

    serial_messenger = Node(
        package="leroi_serial_messenger",
        executable="serial_messenger_node",
        name="serial_messenger_node",
        output="screen",
        parameters=[
            {
                # Leave empty for auto-detect VEX V5 user port.
                "serial_port": "",
                "baud_rate": 115200,
                "read_timeout_sec": 0.05,
                "reconnect_period_sec": 2.0,

                # Serial TX settings
                "tx_append_newline": True,
                "max_payload_bytes": 1024,
                "drop_tx_when_disconnected": True,

                # Heartbeat/status
                "send_heartbeat": True,
                "heartbeat_period_sec": 5.0,
                "status_period_sec": 2.0,

                # Debugging
                "debug_log_rx": False,
                "debug_log_tx": False,
                "debug_log_scan": True,

                # Perception detections can be forwarded to V5 later.
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
                # Input/output topics
                "action_request_topic": "/decision/action_request",
                "robot_command_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",
                "status_topic": "/decision/status",
                "rejected_topic": "/decision/rejected",

                # Command logging
                "command_log_topic": "/decision/command_log",
                "command_log_path": os.path.expanduser(
                    "~/leroiai-jetsonWorkspace/logs/decision_commands.jsonl"
                ),
                "command_history_max": 100,

                # Decision timing
                "status_rate_hz": 2.0,
                "command_timeout_sec": 15.0,

                # Safety rules
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
                # Camera
                "camera_name": "oak_front",
                "camera_type": "oak",
                "camera_index": 0,
                "camera_width": 640,
                "camera_height": 480,
                "camera_fps": 30,
                "oak_mxid": "",

                # Needed for projecting YOLO detections onto the field map.
                # If DepthAI stereo causes issues, temporarily change this to False.
                "oak_with_depth": True,

                # Preview/status
                "preview_rate_hz": 30.0,
                "preview_stream_fps": 15.0,
                "status_rate_hz": 2.0,

                # YOLO TensorRT
                "enable_yolo": True,
                "yolo_rate_hz": 15.0,
                "yolo_model_path": YOLO_MODEL_PATH,
                "yolo_confidence": 0.35,
                "yolo_image_size": 320,
                "yolo_device": "0",
                "yolo_use_half": False,

                # Web preview
                "enable_preview_stream": True,
                "preview_port": 8080,
                "preview_jpeg_quality": 60,
                "draw_detections": True,

                # Map on same 8080 page
                "enable_map_stream": True,
                "map_width_px": 700,
                "map_height_px": 700,

                # VEX field is 12 ft x 12 ft = 144 in.
                # Coordinate convention:
                #   (0, 0) = bottom-left of displayed field map
                #   +X = right
                #   +Y = up/forward
                "field_min_x_in": 0.0,
                "field_max_x_in": 144.0,
                "field_min_y_in": 0.0,
                "field_max_y_in": 144.0,

                # Field map image
                "field_map_image_path": FIELD_MAP_PATH,
                "field_map_rotation": field_map_rotation,
                "field_map_alpha": 0.72,
                "draw_field_grid": True,

                # Robot/object map history
                "robot_trail_max_points": 500,
                "object_history_sec": 3.0,

                # Show route planner suggestion on the preview map.
                "target_path_topic": "/perception/route_suggestion",
                "action_ack_topic": "/robot/action_ack",

                # Camera position relative to robot center.
                # Tune based on actual OAK mounting.
                "camera_offset_x_in": -0.25,  # camera mounted 0.25in left of robot center
                "camera_offset_y_in": 5.91,
                "camera_offset_z_in": 9.84,

                # Disable goal view for now.
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
                # Alliance selection
                "alliance_color": alliance_color,

                # Input/output topics
                "detections_topic": "/perception/oak_front/detections",
                "odom_topic": "/v5/odom",
                "route_suggestion_topic": "/perception/route_suggestion",
                "route_status_topic": "/perception/route_status",

                # Planner rate
                "route_rate_hz": 3.0,
                "stale_detection_timeout_sec": 1.0,
                "stale_odom_timeout_sec": 1.0,

                # Field bounds
                "field_min_x_in": 0.0,
                "field_max_x_in": 144.0,
                "field_min_y_in": 0.0,
                "field_max_y_in": 144.0,

                # Route limits
                "max_route_points": 5,
                "max_carry_blocks": 5,
                "already_held_blocks": 0,

                # Filtering/scoring
                "min_confidence": 0.35,
                "wall_margin_in": 17.72,
                "cluster_dist_in": 35.43,
                "cluster_pickup_dist_in": 14.96,
                "min_route_value": 0.015,

                # Approximate no-pickup zones around field structures.
                # Tune these after seeing real projected detections.
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

                # Debug
                "publish_empty_route": True,
                "debug_log_routes": True,
            }
        ],
    )

    return LaunchDescription(
        [
            alliance_arg,
            field_map_rotation_arg,
            serial_messenger,
            decision,
            perception,
            route_planner,
        ]
    )