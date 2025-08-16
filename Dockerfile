# =================================================================
# 第一阶段: 构建器 (Builder Stage)
# =================================================================
# 使用与最终阶段相同的基础镜像，以确保环境一致性
FROM m.daocloud.io/docker.io/library/python:3.12-slim-bookworm AS builder

# 设置工作目录
WORKDIR /app

# --- 1. 配置 APT 镜像源 (仅用于本阶段) ---
RUN echo "\
Types: deb\n\
URIs: https://mirrors.tuna.tsinghua.edu.cn/debian/\n\
Suites: bookworm bookworm-updates bookworm-backports\n\
Components: main contrib non-free non-free-firmware\n\
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\
" > /etc/apt/sources.list.d/debian.sources

# --- 2. 安装 Python 构建依赖 ---
# 在此阶段我们只需要 pip 和 wheel 工具
RUN pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip wheel

# --- 3. 安装 Python 依赖 ---
# 这是利用缓存的关键步骤：
# 首先只复制依赖定义文件
COPY pyproject.toml ./
# 将所有依赖（包括项目本身）安装到一个临时的虚拟环境中
# 这样做的好处是所有包都被隔离在一个可复制的目录里
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# --no-cache-dir 避免不必要的缓存
RUN pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple .

# =================================================================
# 第二阶段: 最终镜像 (Final Stage)
# =================================================================
# 使用一个全新的、干净的 slim 镜像作为最终运行环境
FROM m.daocloud.io/docker.io/library/python:3.12-slim-bookworm

# 设置工作目录
WORKDIR /app

# --- 1. 配置 APT 镜像源并安装运行时系统依赖 ---
# 这一步是为最终的运行环境安装必要的库
RUN echo "\
Types: deb\n\
URIs: https://mirrors.tuna.tsinghua.edu.cn/debian/\n\
Suites: bookworm bookworm-updates bookworm-backports\n\
Components: main contrib non-free non-free-firmware\n\
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\
" > /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgl1 \
        libgomp1 && \
    # 清理APT缓存，减小镜像体积
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 2. 从构建器阶段复制已安装的 Python 依赖 ---
# (关键步骤) 我们只复制包含所有已安装包的虚拟环境目录
# 这比复制 wheel 文件再重新安装要更直接，且保留了编译好的 C 扩展
COPY --from=builder /opt/venv /opt/venv

# --- 3. 复制应用源代码 ---
COPY . .

# --- 4. 配置运行环境 ---
# 将虚拟环境的 python 和包路径设置为默认
ENV PATH="/opt/venv/bin:$PATH"

# 暴露 FastAPI 服务的端口
EXPOSE 8000

# 容器启动时运行的命令
# uvicorn 会在 /opt/venv/bin/ 中被找到
ENTRYPOINT ["uvicorn", "ocr_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]