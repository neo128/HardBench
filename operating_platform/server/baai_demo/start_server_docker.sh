#!/bin/bash

# ====================== 配置参数（可修改） ======================
CONTAINER_NAME="baai_flask_server"      # 容器名称
IMAGE_NAME="baai-flask-server"  # 镜像名称
#PORTS="-p 8088:8088"                    # 端口映射
ENCODE="-e PYTHONIOENCODING=utf-8"
PORTS="--network host"
PRIVILEGED="--privileged=true"          # 特权模式（谨慎使用）
RESTART_POLICY="--restart unless-stopped" # 重启策略

# 动态获取当前用户名
CURRENT_USER=$(whoami)

# 动态构建卷挂载路径
VOLUMES="-v /home/${CURRENT_USER}/DoRobot/dataset/:/home/robot/dataset/"
VOLUMES2="-v /home/${CURRENT_USER}/Documents/WanX-EI-Studio/operating_platform/server/baai_demo/:/app/code/"
VOLUMES3="-v /home/${CURRENT_USER}/.config/:/home/machine/.config/"

# ====================== 逻辑部分（增强版） ======================

# 检查容器是否存在
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # 检查是否正在运行
    if sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "容器 '${CONTAINER_NAME}' 正在运行。"
        # 检查容器健康状态
        HEALTH=$(sudo docker inspect -f '{{.State.Health.Status}}' ${CONTAINER_NAME} 2>/dev/null)
        if [ "$HEALTH" != "healthy" ]; then
            echo "⚠️ 警告：容器可能处于正在运行！"
            echo "查看日志："
            sudo docker logs --tail 50 ${CONTAINER_NAME}
            read -p "是否重启容器？[y/N] " RESTART_CHOICE
            if [ "$RESTART_CHOICE" == "y" ] || [ "$RESTART_CHOICE" == "Y" ]; then
                sudo docker restart ${CONTAINER_NAME}
                echo "容器已重启"
            fi
        fi
    else
        # 容器存在但未运行
        echo "发现已停止的容器 '${CONTAINER_NAME}'。"
        read -p "是否删除旧容器并创建新容器？[y/N] " DELETE_CHOICE
        if [ "$DELETE_CHOICE" == "y" ] || [ "$DELETE_CHOICE" == "Y" ]; then
            echo "删除旧容器..."
            sudo docker rm -f ${CONTAINER_NAME}
            echo "创建并启动新容器..."
            sudo docker run -d \
                --name ${CONTAINER_NAME} \
                -e LANG=C.UTF-8 \
                -e LC_ALL=C.UTF-8 \
                ${ENCODE} \
                ${PRIVILEGED} \
                ${RESTART_POLICY} \
                ${PORTS} \
                ${VOLUMES} \
                ${VOLUMES2} \
                ${VOLUMES3} \
                ${IMAGE_NAME}
        else
            # 用户选择保留容器
            echo "尝试启动现有容器..."
            sudo docker start ${CONTAINER_NAME}
            echo "容器状态："
            sudo docker ps -a --filter "name=${CONTAINER_NAME}"
        fi
    fi
else
    # 容器不存在，直接创建
    echo "创建并启动新容器 '${CONTAINER_NAME}'..."
    sudo docker run -d \
        --name ${CONTAINER_NAME} \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        ${ENCODE} \
        ${PRIVILEGED} \
        ${RESTART_POLICY} \
        ${PORTS} \
        ${VOLUMES} \
        ${VOLUMES2} \
        ${VOLUMES3} \
        ${IMAGE_NAME}
fi

# 最终状态检查
echo -e "\n最终容器状态："
sudo docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Status}}\t{{.Names}}"
