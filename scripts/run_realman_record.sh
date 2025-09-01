#!/bin/bash

# 配置参数 - 根据实际环境调整
CONTAINER_NAME="operating_platform"
PROJECT_DIR="/root/Operating-Platform"
SCRIPTS_DIR="scripts"
START_SCRIPT="start_realman.sh"
TIMEOUT=60  # 等待超时时间（秒）
CLEANED_UP=false

# 颜色与日志定义
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;32m$1\033[0m"
}

error() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;31m错误: $1\033[0m" >&2
    exit 1
}

# 清理函数 - 处理中断和退出时的资源释放
cleanup() {
    if $CLEANED_UP; then
        log "清理已执行，跳过重复操作"
        return
    fi
    CLEANED_UP=true

    log "开始资源清理..."
    
    # 清理本地PID记录
    if [ -f ".realman_pids" ]; then
        log "终止本地记录的进程..."
        kill $(cat .realman_pids 2>/dev/null) 2>/dev/null || true
        rm -f .realman_pids
    fi

    # 清理容器内可能残留的进程
    if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
        log "清理容器内相关进程..."
        docker exec "$CONTAINER_NAME" pkill -f "$START_SCRIPT" 2>/dev/null || true
        sleep 2
    fi

    # 停止容器（可选：根据需求决定是否停止）
    # if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
    #     log "停止容器 $CONTAINER_NAME..."
    #     docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    # fi

    log "清理完成"
}

# 捕获中断信号 - 确保退出时执行清理
trap cleanup INT TERM EXIT

# 检查Docker是否可用
check_docker() {
    if ! command -v docker &>/dev/null; then
        error "未安装Docker，请先安装Docker"
    fi
    if ! docker info &>/dev/null; then
        error "Docker服务未运行，请启动Docker服务"
    fi
}

# 检查容器是否存在
check_container_exists() {
    if ! docker inspect "$CONTAINER_NAME" &>/dev/null; then
        error "容器 $CONTAINER_NAME 不存在，请先创建容器"
    fi
}

# 等待容器完全就绪
wait_for_container_ready() {
    log "等待容器 $CONTAINER_NAME 就绪..."
    local start_time=$(date +%s)
    
    while true; do
        # 检查容器是否在运行
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
            sleep 2
        # 检查容器内项目目录是否存在
        elif ! docker exec "$CONTAINER_NAME" sh -c "test -d $PROJECT_DIR" 2>/dev/null; then
            sleep 2
        else
            log "容器已就绪"
            return 0
        fi

        # 超时检查
        local current_time=$(date +%s)
        if (( current_time - start_time > TIMEOUT )); then
            error "等待容器就绪超时（${TIMEOUT}秒）"
        fi
    done
}

# 在容器内执行命令并记录日志
execute_in_container() {
    local cmd="$1"
    local log_label="$2"
    local log_file="${log_label}.log"
    
    log "执行容器内命令: $log_label"
    docker exec -t "$CONTAINER_NAME" bash -c "$cmd" \
        > >(tee -a "$log_file") 2>&1 &
    echo $! >> .realman_pids
    log "命令进程已启动，日志文件: $log_file"
}

# 主程序执行流程
main() {
    log "===== 开始启动Realman程序 ====="

    # 前置检查
    check_docker
    check_container_exists

    # 停止并重启容器（确保干净启动）
    log "停止现有容器实例..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    sleep 2

    log "启动容器 $CONTAINER_NAME..."
    if ! docker start "$CONTAINER_NAME"; then
        error "容器启动失败"
    fi

    # 等待容器就绪
    wait_for_container_ready

    # 清理旧的日志和PID文件
    rm -f .realman_pids
    rm -f dataflow.log coordinator.log

    # 构造容器内命令序列
    local full_script_path="${PROJECT_DIR}/${SCRIPTS_DIR}/${START_SCRIPT}"
    local container_commands="cd ${PROJECT_DIR}/${SCRIPTS_DIR} && bash ${START_SCRIPT}"

    # 执行容器内启动命令
    execute_in_container "$container_commands" "realman_start"

    log "所有启动步骤已完成"
    log "程序运行中，按 Ctrl+C 停止并清理资源"
    
    # 保持脚本运行以监控进程
    wait
}

# 启动主程序
main
