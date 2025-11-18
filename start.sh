#!/bin/bash

# Xcode AI Proxy 启动脚本
# 确保脚本在错误时停止
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Xcode AI Proxy 启动脚本${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# 检查 Python 版本
echo -e "${YELLOW}[1/5] 检查 Python 版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3，请先安装 Python 3.12+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python 版本: ${PYTHON_VERSION}${NC}"
echo ""

# 检查 uv 是否安装
echo -e "${YELLOW}[2/5] 检查 uv 包管理工具...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: 未找到 uv，请先安装 uv${NC}"
    echo "安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}✓ uv 已安装${NC}"
echo ""

# 检查配置文件
echo -e "${YELLOW}[3/5] 检查配置文件...${NC}"
if [ ! -f "models.toml" ]; then
    echo -e "${RED}错误: 未找到 models.toml 配置文件${NC}"
    exit 1
fi
echo -e "${GREEN}✓ models.toml 配置文件存在${NC}"
echo ""

# 提示用户配置 API 密钥
echo -e "${YELLOW}[4/5] 检查 API 密钥配置...${NC}"
echo -e "${YELLOW}提示: 请确保已在 models.toml 文件中配置至少一个供应商的 API 密钥${NC}"
echo "查看配置文件: cat models.toml"
echo "编辑配置文件: vim models.toml"
echo -e "${GREEN}✓ API 密钥配置检查完成${NC}"
echo ""

# 创建虚拟环境（如果不存在）
echo -e "${YELLOW}[5/5] 配置 Python 虚拟环境...${NC}"
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    uv venv
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
else
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
fi

# 激活虚拟环境并安装依赖
echo "安装依赖..."
source .venv/bin/activate
uv sync
echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 启动服务
echo -e "${YELLOW}[5/5] 启动 Xcode AI Proxy 服务...${NC}"
echo -e "${GREEN}服务地址: http://localhost:8899${NC}"
echo -e "${GREEN}按 Ctrl+C 停止服务${NC}"
echo ""
echo -e "${GREEN}======================================${NC}"
echo ""

# 启动 Python 服务
python3 server.py
