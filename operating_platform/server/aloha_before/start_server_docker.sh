#!/bin/bash

# ====================== 配置参数（可修改） ======================
CONTAINER_NAME="baai_flask_server"      # 容器名称
IMAGE_NAME="baai-flask-server-release"  # 镜像名称
PORTS="-p 8080:8080"                    # 端口映射（主机端口:容器端口）
PRIVILEGED="--privileged=true"          # 特权模式（谨慎使用）
RESTART_POLICY="--restart unless-stopped" # 重启策略

# 动态获取当前用户名（兼容性更好）
CURRENT_USER=$(whoami)  # 或使用 $USER

# 动态构建卷挂载路径（替换原硬编码的 "agilex"）
VOLUMES="-v /home/${CURRENT_USER}/Documents/Operating-Platform/:/home/agilex/Documents/Ryu-Yang/Operating-Platform/"
VOLUMES2="-v /home/${CURRENT_USER}/Documents/server/release/Operating-Platform/operating_platform/server/:/app/code/"
VOLUMES3="-v /home/${CURRENT_USER}/.config/:/home/machine/.config/"

# ====================== 逻辑部分（无需修改） ======================

# 检查容器是否存在
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # 容器存在，检查是否正在运行
    if sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "容器 '${CONTAINER_NAME}' 已经在运行，无需操作。"
    else
        # 容器存在但未运行，启动它
        echo "启动已存在的容器 '${CONTAINER_NAME}'..."
        sudo docker start ${CONTAINER_NAME}
    fi
else
    # 容器不存在，创建并运行
    echo "创建并启动新容器 '${CONTAINER_NAME}'..."
    sudo docker run -d \
        --name ${CONTAINER_NAME} \
        ${PRIVILEGED} \
        ${RESTART_POLICY} \
        ${PORTS} \
        ${VOLUMES} \
        ${VOLUMES2} \
        ${VOLUMES3} \
        ${IMAGE_NAME}
fi

# 检查容器状态
echo "当前容器状态："
sudo docker ps -a --filter "name=${CONTAINER_NAME}" --format 'table {{.Names}}\t{{.Status}}'