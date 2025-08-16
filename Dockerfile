# my-ocr-project/backend/Dockerfile

# --- Build Stage ---
# 使用一个包含构建工具的镜像来安装依赖
FROM m.daocloud.io/docker.io/library/python:3.12-slim-bookworm AS builder

WORKDIR /app

# 安装 uv，这是一个现代化的、快速的 Python 包管理器
RUN pip install uv

# 复制依赖定义文件
COPY uv.lock .python-version pyproject.toml .

# 使用 uv 安装依赖到系统 Python 环境中，方便下一阶段复制
# 利用 Docker 缓存，只有 pyproject.toml 变化时才重新安装
RUN uv sync --frozen --no-cache

# --- Production Stage ---
# 使用一个干净的镜像作为最终的生产镜像
FROM m.daocloud.io/docker.io/library/python:3.12-slim-bookworm AS production

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /app/.venv/ /app/.venv/
COPY --from=builder /usr/local/bin /usr/local/bin

# 安装 opencv 的系统级依赖，这是运行 paddleocr 所必需的
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 复制应用源代码
COPY ./ocr_gateway ./ocr_gateway

# 暴露 FastAPI 服务的端口
EXPOSE 8000

# 容器启动时运行的命令
CMD ["/app/.venv/bin/fastapi", "run", "ocr_gateway/main.py", "--port", "8000", "--host", "0.0.0.0"]