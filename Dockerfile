# --- Build Stage ---
FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装 uv
RUN pip install uv

# 复制依赖定义文件
COPY pyproject.toml .

# 使用 uv 安装依赖，利用缓存
# --system 会将包装到系统python中，方便下一阶段复制
RUN uv pip install --no-cache --system -r pyproject.toml

# --- Production Stage ---
FROM python:3.11-slim as production

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 安装 opencv 的系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY ./ocr_gateway ./ocr_gateway

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "ocr_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]