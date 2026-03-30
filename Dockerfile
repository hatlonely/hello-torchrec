# Ubuntu 22.04 基础镜像
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 添加 deadsnakes PPA 并安装 Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接，使 python 命令可用
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# 升级 pip
RUN python -m pip install --upgrade pip

# 安装 PyTorch (CPU 版本)
# 使用与 requirements.txt 相同的版本
RUN pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 FBGEMM (CPU 版本)
RUN pip install fbgemm-gpu==1.0.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 TorchRec (CPU 版本)
RUN pip install torchrec==1.0.0 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
# 注意：networkx 3.6.1 需要 Python 3.12+，使用 3.4.2 兼容 Python 3.11
RUN pip install \
    torchmetrics==1.0.3 \
    iopath==0.1.10 \
    pyre-extensions==0.0.32 \
    tqdm==4.67.3 \
    numpy==1.26.4 \
    networkx==3.4.2 \
    PyYAML

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 生成训练数据（可选，也可以在运行时生成）
RUN python -m sample.main --num-samples 10000 --output-path data/samples.json

# 暴露端口（多节点训练可能需要）
EXPOSE 29500

# 默认命令：显示帮助信息
CMD ["python", "-m", "train.main", "--help"]
