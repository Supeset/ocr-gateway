# ocr_gateway/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .core.config import settings
from .api.v1 import recognize
from .services import ocr_service

# --- 核心修改 1: 实现 lifespan 事件管理器 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在应用启动时加载模型，在应用关闭时清理。
    """
    # === 应用启动时执行 ===
    # 将加载好的模型引擎存储在 app.state 中，以便在请求中全局访问
    app.state.ocr_engines = ocr_service.load_ocr_engines()
    app.state.openai_client = ocr_service.get_openai_client()
    
    yield # lifespan 的分界点
    
    # === 应用关闭时执行 ===
    app.state.ocr_engines.clear()
    logging.info("OCR 引擎已清理。")

# --- 核心修改 2: 将 lifespan 应用到 FastAPI 实例 ---
app = FastAPI(
    title=settings.APP_NAME,
    description="一个使用 FastAPI 和 PaddleOCR 的证照识别 API 网关",
    version="1.0.0",
    lifespan=lifespan # <-- 在这里应用
)

# 挂载 v1 版本的路由
app.include_router(recognize.router, prefix=settings.API_V1_STR, tags=["Recognition"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"欢迎使用 {settings.APP_NAME}"}