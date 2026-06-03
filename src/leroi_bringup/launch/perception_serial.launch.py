from launch import LaunchDescription
from launch_ros.actions import Node


YOLO_MODEL_PATH = "/home/leroi-ultio/leroiai-jetsonWorkspace/models/yolo/best_fp16.engine"


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

                "tx_append_newline": True,
                "max_payload_bytes": 1024,
                "drop_tx_when_disconnected": True,

                "send_heartbeat": True,
                "heartbeat_period_sec": 1.0,
                "status_period_sec": 1.0,

                # Keep scan/tx readable, but avoid RX spam unless debugging.
                "debug_log_rx": False,
                "debug_log_tx": True,
                "debug_log_scan": True,

                # Perception detections can be forwarded to V5 later.
                "detection_topic": "/perception/oak_front/detections",
                "forward_detections_to_v5": False,
                "debug_log_detections": False,
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
                "oak_with_depth": False,

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
                "field_min_x_cm": -183.0,
                "field_max_x_cm": 183.0,
                "field_min_y_cm": -183.0,
                "field_max_y_cm": 183.0,
                "robot_trail_max_points": 500,
                "object_history_sec": 3.0,
                "target_path_topic": "/robot/path_command",
                "action_ack_topic": "/robot/action_ack",

                # Disable goal view for now
                "enable_goal_view": False,
            }
        ],
    )

    return LaunchDescription(
        [
            serial_messenger,
            perception,
        ]
    )