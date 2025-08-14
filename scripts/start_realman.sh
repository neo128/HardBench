#!/usr/bin/env bash
set -e

# # ---------- 自动更新 ----------
# echo ">>> Checking upstream for updates..."
# # 切到仓库根目录再操作，确保脚本在哪儿都能运行
# REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # 假设脚本放在 repo 根目录
# cd "$REPO_DIR"

# # 拉最新代码（非交互、不提示 host key）
# git config --global advice.detachedHead false
# git remote set-url origin https://github.com/XuRuntian/Operating-Platform.git
# git fetch --depth=1 origin
# LOCAL=$(git rev-parse HEAD)
# REMOTE=$(git rev-parse @{u})

# if [[ "$LOCAL" != "$REMOTE" ]]; then
#     echo ">>> New commits detected, pulling..."
#     git pull --ff-only origin "$(git rev-parse --abbrev-ref HEAD)"
#     echo ">>> Repository updated."
# else
#     echo ">>> Already up-to-date."
# fi

# ---------- 原有启动逻辑 ----------
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. 启动 dora
cd ../operating_platform/robot/robots/realman_v1
conda activate op-robot-realman
dora run robot_realman_dataflow.yml &
DORA_PID=$!

# 2. 启动 coordinator
cd  ../../../core/
conda activate op
pip install Robotic_Arm
exec python coordinator.py --robot.type=realman































