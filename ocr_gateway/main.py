# ocr_gateway/main.py
from fastapi import FastAPI
from .core.config import settings
from .api.v1 import recognize

app = FastAPI(
    title=settings.APP_NAME,
    description="一个使用 FastAPI 和 PaddleOCR 的证照识别 API 网关",
    version="1.0.0"
)

# 挂载 v1 版本的路由
app.include_router(recognize.router, prefix=settings.API_V1_STR, tags=["Recognition"])

@app.get("/", tags=["Root"])
def read_root():
    """
    根路径，用于健康检查。
    """
    return {"message": f"欢迎使用 {settings.APP_NAME}"}