# ocr_gateway/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # --- App Settings ---
    APP_NAME: str = "OCR Gateway"
    API_V1_STR: str = "/api/v1"

    # --- OCR Settings ---
    OCR_USE_GPU: bool = False
    OCR_LANG: str = "ch"

    # --- AI Settings (loaded from .env) ---
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = "https://openrouter.ai/api/v1"
    OPENAI_MODEL_NAME: str = "deepseek/deepseek-chat-v3-0324:free"
    # 用于 OpenRouter 的额外 HTTP 头信息
    OPENROUTER_REFERER: str = "http://localhost:3000" # 最好换成你前端的地址
    OPENROUTER_TITLE: str = "OCR Gateway"

settings = Settings()