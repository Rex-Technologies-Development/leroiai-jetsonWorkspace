import json
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


# Command scenario the test driver walks through, one ACK at a time.
#
#   1) MOVE_CHASSIS_ONLY (action_id 1) -> first target point.
#   2) MOVE_CHASSIS_ONLY (action_id 1) -> second target point.
#   3) COLLECT_BLOCKS    (action_id 2) -> a bunch of target points.
#
# Field is 365.76 cm x 365.76 cm, origin bottom-left. Edit freely.
TEST_SEQUENCE = [
    {
        "action_id": 1,
        "points": [{"x": 120.0, "y": 90.0, "heading": 0.0}],
        "params": {"label": "move_to_point_1"},
    },
    {
        "action_id": 1,
        "points": [{"x": 240.0, "y": 180.0, "heading": 90.0}],
        "params": {"label": "move_to_point_2"},
    },
    {
        "action_id": 2,
        "points": [
            {"x": 250.0, "y": 200.0, "heading": 90.0},
            {"x": 270.0, "y": 220.0, "heading": 45.0},
            {"x": 290.0, "y": 240.0, "heading": 0.0},
            {"x": 300.0, "y": 260.0, "heading": -45.0},
        ],
        "params": {"label": "collect_blocks_run"},
    },
]


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

                # Web preview + map (same 8080 page)
                "enable_preview_stream": True,
                "preview_port": 8080,
                "preview_jpeg_quality": 60,
                "draw_detections": True,
                "enable_map_stream": True,
                "map_width_px": 700,
                "map_height_px": 700,

                # Field bounds
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

                # Show approved command path on the map.
                "target_path_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",

                "camera_offset_x_cm": -0.635,  # camera mounted 0.25in left of robot center
                "camera_offset_y_cm": 15.0,
                "camera_offset_z_cm": 25.0,

                "enable_goal_view": False,
            }
        ],
    )

    test_driver = Node(
        package="leroi_decision",
        executable="decision_test_driver",
        name="decision_test_driver",
        output="screen",
        parameters=[
            {
                "action_request_topic": "/decision/action_request",
                "robot_command_topic": "/robot/command",
                "action_ack_topic": "/robot/action_ack",

                # Wait this long after launch before sending command 1.
                "startup_delay_sec": LaunchConfiguration("startup_delay_sec"),
                # How long the simulated robot "works" before ACKing DONE.
                "sim_ack_delay_sec": 1.0,
                # Gap after an ACK before sending the next command.
                "inter_command_delay_sec": 1.0,
                # True: this node fakes the robot ACKs. Set false if a real
                # robot/serial bridge is connected and will ACK on its own.
                "simulate_robot_ack": LaunchConfiguration("simulate_robot_ack"),
                "loop_sequence": False,

                "sequence_json": json.dumps(TEST_SEQUENCE),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "startup_delay_sec",
                default_value="5.0",
                description="Seconds to wait after launch before the first command",
            ),
            DeclareLaunchArgument(
                "simulate_robot_ack",
                default_value="false",
                description="Have the test driver fake robot ACKs. Default false: "
                "the real robot/serial bridge ACKs. Set true to test without a robot.",
            ),
            serial_messenger,
            perception,
            decision,
            test_driver,
        ]
    )
