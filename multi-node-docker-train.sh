#!/bin/bash
# 单机多容器分布式训练脚本
# 在同一台机器上启动 3 个 Docker 容器进行多节点训练

# 确保使用 bash 执行
if [ -z "$BASH_VERSION" ]; then
    echo "错误: 此脚本需要使用 bash 执行"
    echo "请使用: bash $0 或 ./$0"
    exit 1
fi

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
NETWORK_NAME="torchrec-network"
NUM_NODES=3
NPROC_PER_NODE=2
CONFIG_FILE="config/distributed.yaml"
MASTER_PORT=29500
CONTAINER_PREFIX="torchrec-node"

# 打印使用说明
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "在单机上启动多个 Docker 容器进行多节点分布式训练"
    echo ""
    echo "Options:"
    echo "  --nodes N              节点数量 (默认: 3)"
    echo "  --nproc N              每个节点的进程数 (默认: 2)"
    echo "  --config FILE          配置文件 (默认: config/distributed.yaml)"
    echo "  --port PORT            master 端口 (默认: 29500)"
    echo "  --clean                清理容器和网络后退出"
    echo "  --help                 显示帮助信息"
    echo ""
    echo "Example:"
    echo "  $0                    # 使用默认配置启动 3 个节点"
    echo "  $0 --nodes 2 --nproc 4  # 启动 2 个节点，每个节点 4 个进程"
    echo "  $0 --clean            # 清理所有容器"
}

# 清理函数
cleanup() {
    echo -e "${YELLOW}清理容器和网络...${NC}"

    # 停止并删除所有容器
    for i in $(seq 0 $((NUM_NODES - 1))); do
        container_name="${CONTAINER_PREFIX}${i}"
        if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
            echo "停止容器: ${container_name}"
            docker stop "${container_name}" 2>/dev/null || true
            docker rm "${container_name}" 2>/dev/null || true
        fi
    done

    # 删除网络
    if docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
        echo "删除网络: ${NETWORK_NAME}"
        docker network rm "${NETWORK_NAME}" 2>/dev/null || true
    fi

    echo -e "${GREEN}清理完成${NC}"
}

# 检查 Docker 镜像
check_image() {
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^hello-torchrec:latest$"; then
        echo -e "${RED}错误: Docker 镜像 hello-torchrec:latest 不存在${NC}"
        echo "请先运行: docker build -t hello-torchrec:latest ."
        exit 1
    fi
}

# 检查配置文件
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
        exit 1
    fi
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --clean)
            cleanup
            exit 0
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# 设置陷阱，确保在退出时清理
trap cleanup EXIT INT TERM

# 检查镜像和配置
check_image
check_config

# 打印配置信息
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}单机多容器分布式训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "节点数量:        ${NUM_NODES}"
echo -e "每节点进程数:    ${NPROC_PER_NODE}"
echo -e "配置文件:        ${CONFIG_FILE}"
echo -e "Master 端口:     ${MASTER_PORT}"
echo -e "网络名称:        ${NETWORK_NAME}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 创建 Docker 网络
echo -e "${YELLOW}创建 Docker 网络...${NC}"
if docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
    echo "网络 ${NETWORK_NAME} 已存在"
else
    docker network create --driver bridge "${NETWORK_NAME}"
    echo -e "${GREEN}✓ 网络创建成功${NC}"
fi

# 启动容器
echo ""
echo -e "${YELLOW}启动 ${NUM_NODES} 个容器...${NC}"

# 创建数组存储容器 PID
declare -a container_pids

for i in $(seq 0 $((NUM_NODES - 1))); do
    container_name="${CONTAINER_PREFIX}${i}"

    echo -e "${GREEN}启动容器 ${container_name} (rank ${i})...${NC}"

    # 启动容器
    # 使用 --init 防止僵尸进程
    # 使用 --network 连接到同一网络
    # 使用 --hostname 设置主机名
    # node0 作为 master，使用容器名作为地址
    docker run -d \
        --name "${container_name}" \
        --network "${NETWORK_NAME}" \
        --hostname "${container_name}" \
        --init \
        -v "$(pwd)/outputs:/app/outputs" \
        -v "$(pwd)/data:/app/data" \
        -e PYTHONUNBUFFERED=1 \
        hello-torchrec:latest \
        torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --master_addr="${CONTAINER_PREFIX}0" \
        --master_port=${MASTER_PORT} \
        --node_rank=${i} \
        -m train.main \
        --config "${CONFIG_FILE}" > /dev/null

    echo -e "${GREEN}✓ 容器 ${container_name} 启动成功${NC}"
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}所有容器已启动${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "查看实时日志:"
echo -e "  ${YELLOW}docker logs -f ${CONTAINER_PREFIX}0${NC}  # master 节点"
echo -e "  ${YELLOW}docker logs -f ${CONTAINER_PREFIX}1${NC}  # worker 节点 1"
echo -e "  ${YELLOW}docker logs -f ${CONTAINER_PREFIX}2${NC}  # worker 节点 2"
echo ""
echo -e "查看所有日志:"
echo -e "  ${YELLOW}docker-compose logs${NC}"
echo ""
echo -e "查看容器状态:"
echo -e "  ${YELLOW}docker ps${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}训练进行中...${NC}"
echo -e "${YELLOW}按 Ctrl+C 停止训练并清理容器${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示 master 节点的日志
docker logs -f "${CONTAINER_PREFIX}0"
