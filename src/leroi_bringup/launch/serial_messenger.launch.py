from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    serial_messenger = Node(
        package="leroi_serial_messenger",
        executable="serial_messenger_node",
        name="serial_messenger_node",
        output="screen",
        parameters=[
            {
                # Leave empty for auto-detect.
                "serial_port": "",
                "baud_rate": 115200,
                "read_timeout_sec": 0.05,
                "reconnect_period_sec": 2.0,

                "tx_append_newline": True,
                "max_payload_bytes": 512,

                "send_heartbeat": True,
                "heartbeat_period_sec": 1.0,
                "status_period_sec": 1.0,

                "debug_log_rx": False,
                "debug_log_tx": True,

                "drop_tx_when_disconnected": True,
            }
        ],
    )

    return LaunchDescription([
        serial_messenger,
    ])