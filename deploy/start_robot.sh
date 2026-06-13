#!/usr/bin/env bash
# Launch the default on-robot stack (integration policy test, autonomous).
# Used both for manual runs and by the leroi-robot systemd service.
# NOTE: no `set -u` — ROS/colcon setup files reference unbound vars and would
# abort the script (and crash-loop the service) under `set -u`.
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/leroi-ultio/leroiai-jetsonWorkspace/install/setup.bash

# The perception node's YOLO detector imports torch, whose CUDA deps
# (libcudss.so.0 etc.) ship in the pip nvidia-cu12 package. torch's RUNPATH is
# only [$ORIGIN], so those libs are found ONLY via LD_LIBRARY_PATH. Interactive
# shells get this from ~/.bashrc, but systemd does not source ~/.bashrc — so
# without this the perception node dies at startup with
# "ImportError: libcudss.so.0" (surfaced misleadingly as "ultralytics is not
# installed"). Mirror the ~/.bashrc CUDA paths so the service matches the shell.
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/home/leroi-ultio/.local/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH}"

exec ros2 launch leroi_bringup robot.launch.py
