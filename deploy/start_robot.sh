#!/usr/bin/env bash
# Launch the default on-robot stack (integration policy test, autonomous).
# Used both for manual runs and by the leroi-robot systemd service.
# NOTE: no `set -u` — ROS/colcon setup files reference unbound vars and would
# abort the script (and crash-loop the service) under `set -u`.
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/leroi-ultio/leroiai-jetsonWorkspace/install/setup.bash

exec ros2 launch leroi_bringup robot.launch.py
