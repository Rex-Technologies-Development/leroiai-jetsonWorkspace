from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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
            Node(
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
                        "action_ack_topic": "/robot/action_ack",
                        "status_topic": "/dashboard/status",
                        "sync_rate_hz": 2.0,
                        "reconnect_period_sec": 5.0,
                        "field_cm_to_dashboard": 1.0,
                        "enable_robot_channel": True,
                        "enable_blocks_channel": True,
                        "enable_camera_channel": True,
                        "camera_id": "cam-1",
                        "camera_label": "oak_front",
                        "camera_stream_fps": 10.0,
                        "camera_preview_url": "http://127.0.0.1:8080/camera.jpg",
                    }
                ],
            ),
        ]
    )
