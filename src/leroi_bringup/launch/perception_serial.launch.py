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

                # Important for projecting YOLO detections onto the map.
                # If this causes DepthAI issues, set it back to False temporarily.
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

                # VEX field is 12 ft x 12 ft = 365.76 cm.
                # Coordinate convention:
                #   (0, 0) = bottom-left of the displayed field map
                #   +X = right
                #   +Y = up/forward
                "field_min_x_cm": 0.0,
                "field_max_x_cm": 365.76,
                "field_min_y_cm": 0.0,
                "field_max_y_cm": 365.76,

                # Field map image
                "field_map_image_path": FIELD_MAP_PATH,

                # Options:
                #   "none" = original image orientation
                #   "cw"   = rotate clockwise
                #   "ccw"  = rotate counterclockwise
                #   "180"  = rotate 180 degrees
                #
                # Use "cw" or "ccw" depending on which color parking square
                # you want on the bottom edge.
                "field_map_rotation": "cw",
                "field_map_alpha": 0.72,
                "draw_field_grid": True,

                # Robot/object map history
                "robot_trail_max_points": 500,
                "object_history_sec": 3.0,

                # The decision layer now publishes approved commands to /robot/command.
                # The perception map can read this and display the target path.
                "target_path_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",

                # Camera position relative to robot center.
                # Tune these later based on where the OAK camera is mounted.
                "camera_offset_x_cm": -0.635,  # camera mounted 0.25in left of robot center
                "camera_offset_y_cm": 15.0,
                "camera_offset_z_cm": 25.0,

                # Disable goal view for now
                "enable_goal_view": False,
            }
        ],
    )

    dashboard = Node(
        package="leroi_dashboard",
        executable="dashboard_node",
        name="dashboard_node",
        output="screen",
        parameters=[
            {
                "enable_dashboard": LaunchConfiguration("enable_dashboard"),
                "bot_id": LaunchConfiguration("dashboard_bot_id"),
                "vexai_host": LaunchConfiguration("vexai_host"),
                "odom_topic": "/v5/odom",
                "detection_topic": "/perception/oak_front/detections",
                "mapped_objects_topic": "/perception/oak_front/mapped_objects",
                "action_ack_topic": "/robot/action_ack",
                "status_topic": "/dashboard/status",
                "sync_rate_hz": 2.0,
                "reconnect_period_sec": 5.0,
                "field_cm_to_dashboard": 1.0,
                "send_all_field_detections": True,
                "block_class_substrings": ["block", "red", "blue"],
                "object_history_sec": 3.0,
                "camera_offset_x_cm": -0.635,  # camera mounted 0.25in left of robot center
                "camera_offset_y_cm": 15.0,
                "camera_offset_z_cm": 25.0,
                "enable_robot_channel": True,
                "enable_blocks_channel": True,
                "enable_camera_channel": True,
                "camera_id": "cam-1",
                "camera_label": "oak_front",
                "camera_stream_fps": 10.0,
                "camera_preview_url": "http://127.0.0.1:8080/camera.jpg",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_dashboard",
                default_value="true",
                description="Try to sync with VexAI dashboard; robot runs offline if unreachable",
            ),
            DeclareLaunchArgument(
                "dashboard_bot_id",
                default_value="leroi-bot",
                description="VexAI dashboard bot ID",
            ),
            DeclareLaunchArgument(
                "vexai_host",
                default_value="162.248.101.133:6741",
                description="VexAI dashboard WebSocket host",
            ),
            serial_messenger,
            decision,
            perception,
            dashboard,
        ]
    )