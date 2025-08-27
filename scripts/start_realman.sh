#!/usr/bin/env bash
set -e
# 需要提前进行的操作 git config --global --add safe.directory "$(pwd)"   
source "$(conda info --base)/etc/profile.d/conda.sh"
# 1. 启动 dora
cd ../operating_platform/robot/robots/realman_v1
conda activate op-robot-realman
dora run robot_realman_dataflow.yml &
DORA_PID=$!

# 2. 启动 coordinator
cd  ../../../core/
conda activate op
exec python coordinator.py --robot.type=realman
































