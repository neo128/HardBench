#!/usr/bin/env bash
set -e

# 0. 让脚本能找到 conda
source "$(conda  info --base)/etc/profile.d/conda.sh"

# 1. 启动 dora（环境：op-robot-realman）
cd ~/Operating-Platform/operating_platform/robot/robots/realman_v1
conda activate op-robot-realman
dora run robot_realman_dataflow.yml &
DORA_PID=$!

# 2. 启动 coordinator（环境：op）
cd ~/Operating-Platform/operating_platform/core
conda activate op
exec python coordinator.py --robot.type=realman