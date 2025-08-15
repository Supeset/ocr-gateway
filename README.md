# OCR Gateway

这是一个使用 FastAPI 和 PaddleOCR 构建的证照识别 API 服务。

## 功能

-   通过 HTTP API 上传图片 (`.jpg`, `.png`)。
-   对图片中的文字进行识别。
-   以 JSON 格式返回识别结果，包括文字内容、位置坐标和置信度。

## 技术栈

-   **Web 框架**: [FastAPI](https://fastapi.tiangolo.com/)
-   **包管理器**: [uv](https://github.com/astral-sh/uv)
-   **OCR 引擎**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 如何运行

### 1. 环境准备

-   确保你已经安装了 Python 3.9+。
-   安装 `uv` 包管理器:
    ```bash
    pip install uv
    ```

### 2. 安装依赖

在项目根目录下，使用 `uv` 创建虚拟环境并安装所有依赖。

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境 (macOS/Linux)
source .venv/bin/activate

# 激活虚拟环境 (Windows)
# .\.venv\Scripts\activate

# 安装依赖
uv pip install -r pyproject.toml

uvicorn ocr_gateway.main:app --reload