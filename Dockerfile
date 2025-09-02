# file: Dockerfile

# 使用一个稳定的 Ubuntu 基础镜像
FROM ubuntu:22.04

# 设置环境变量，避免 apt-get 在构建过程中交互
ENV DEBIAN_FRONTEND=noninteractive

# 更新包列表并安装通用工具和 Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    tree \
    curl \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 将 python3 设置为 python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 更换 pip 源为华为云镜像
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/
# 信任该源，避免 HTTPS 警告
RUN pip config set global.trusted-host repo.huaweicloud.com

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip

# 创建工作目录
WORKDIR /app

# 复制 Agent 的依赖需求文件
COPY requirements.txt .

# 安装 Agent 的 Python 依赖，并预装 pytest 和 hunter
RUN pip install --no-cache-dir -r requirements.txt pytest hunter

# 复制整个 Agent 代码到工作目录
COPY runnable_agent_batch/ ./runnable_agent_batch/

# 设置一个默认的命令，方便调试
CMD ["/bin/bash"]