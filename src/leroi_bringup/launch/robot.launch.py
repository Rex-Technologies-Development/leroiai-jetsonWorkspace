from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Main robot entry point.

    Delegates to the integration policy test stack (perception + serial +
    route planner + decision + policy), which is the default on-robot run:
    autonomous (no confirmation), forced COLLECT -> SCORE_LONG loop for 2
    cycles, then STOP. Override any of its launch args here if needed.

    Also starts the telemetry logger, which records every policy / decision /
    movement / perception message to logs/ros2_telemetry.jsonl. This is the
    on-disk replacement for `ros2 topic echo`, which is not usable on the robot
    (headless systemd run). The live camera + field map remain viewable at
    http://<jetson>:8080 (served by the perception node).
    """
    launch_dir = os.path.join(
        get_package_share_directory("leroi_bringup"),
        "launch",
    )

    telemetry_node = Node(
        package="leroi_telemetry",
        executable="telemetry_node",
        name="telemetry_node",
        output="screen",
        # Append across runs and keep the history in git. Each run is delimited
        # by a RUN_START banner line, so old runs stay readable.
        parameters=[{"truncate_on_start": False}],
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_dir, "integration_policy_test.launch.py"),
                ),
            ),
            telemetry_node,
        ]
    )
