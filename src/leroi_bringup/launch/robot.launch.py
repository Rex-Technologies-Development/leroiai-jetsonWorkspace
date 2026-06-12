from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Main robot entry point.

    Delegates to the integration policy test stack (perception + serial +
    route planner + decision + policy), which is the default on-robot run:
    autonomous (no confirmation), forced COLLECT -> SCORE_LONG loop for 2
    cycles, then STOP. Override any of its launch args here if needed.
    """
    launch_dir = os.path.join(
        get_package_share_directory("leroi_bringup"),
        "launch",
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_dir, "integration_policy_test.launch.py"),
                ),
            ),
        ]
    )
