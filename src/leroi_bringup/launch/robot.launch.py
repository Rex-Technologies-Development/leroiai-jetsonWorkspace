from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    perception_node = Node(
        package="leroi_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {
                "publish_rate_hz": 2.0,
                "enable_fake_detections": True,
            }
        ],
    )

    return LaunchDescription([
        perception_node,
    ])