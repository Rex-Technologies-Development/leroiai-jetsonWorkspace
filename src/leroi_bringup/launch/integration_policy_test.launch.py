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
                "camera_offset_x_cm": 0.0,
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
                "odom_topic": "/v5/odom",
                "decision_status_topic": "/decision/status",

                # Output + operator gate
                "action_request_topic": "/decision/action_request",
                "pending_action_topic": "/policy/pending_action",
                "decision_topic": "/policy/decision",
                "status_topic": "/policy/status",

                # Model
                "model_path": LaunchConfiguration("policy_model_path"),
                # CPU is plenty for a small policy MLP and keeps the GPU free for
                # perception's YOLO TensorRT engine. To use the GPU, install the
                # Jetson onnxruntime-gpu build and set e.g.:
                # "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
                "onnx_providers": "CPUExecutionProvider",
                "obs_dim_override": 0,
                "action_id_map": "",

                # Behavior
                "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
                "require_confirmation": LaunchConfiguration("require_confirmation"),
                "respect_decision_busy": True,
                "max_blocks": 10,
                "field_size_cm": 365.76,
            }
        ],
    )

    return LaunchDescription(
        [
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
            decision,
            policy,
        ]
    )
