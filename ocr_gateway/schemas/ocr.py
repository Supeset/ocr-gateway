# ocr_gateway/schemas/ocr.py
from pydantic import BaseModel, Field
from typing import List, Optional

class RecognitionItem(BaseModel):
    box: List[List[int]]
    text: str
    confidence: float

class OCRResponse(BaseModel):
    filename: str
    data: List[RecognitionItem] = Field(..., description="识别出的文本项列表")
    summary: Optional[str] = Field(None, description="由 AI 生成的结构化信息总结 (JSON 字符串)")