#!/usr/bin/env bash
set -e
# 优先选择NVIDIA硬件GPU（对Rerun渲染至关重要）
export VK_DEVICE_SELECTOR="vendorId=0x10de"
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
































