# ocr_gateway/services/ocr_service.py
import logging
from functools import lru_cache
from typing import Dict, Any

import cv2
import numpy as np
from openai import OpenAI
# -------------------- 核心修改 1: 导入独立的检测和识别模块 --------------------
from paddleocr import TextDetection, TextRecognition

from ..core.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 核心修改 2: 创建独立的引擎初始化函数 --------------------
@lru_cache(maxsize=1)
def get_detection_engine() -> TextDetection | None:
    """初始化并返回一个单例的文本检测引擎。"""
    try:
        logger.warning("Initializing Text Detection engine...")
        # 使用默认的服务器端检测模型
        engine = TextDetection(device='gpu' if settings.OCR_USE_GPU else 'cpu')
        logger.warning("Text Detection engine initialized successfully.")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize Text Detection engine: {e}", exc_info=True)
        return None

@lru_cache(maxsize=1)
def get_recognition_engine() -> TextRecognition | None:
    """初始化并返回一个单例的文本识别引擎。"""
    try:
        logger.warning("Initializing Text Recognition engine (PP-OCRv5_server_rec)...")
        # 明确使用文档中推荐的最新、最强大的 v5 服务器模型
        engine = TextRecognition(
            model_name="PP-OCRv5_server_rec",
            device='gpu' if settings.OCR_USE_GPU else 'cpu'
        )
        logger.warning("Text Recognition engine initialized successfully.")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize Text Recognition engine: {e}", exc_info=True)
        return None


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI | None:
    """
    初始化并返回一个单例的 OpenAI 客户端。(此函数无需修改)
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found. AI summarization will be disabled.")
        return None
    try:
        logger.warning(f"Initializing OpenAI client for model '{settings.OPENAI_MODEL_NAME}'.")
        client = OpenAI(base_url=settings.OPENAI_BASE_URL, api_key=settings.OPENAI_API_KEY)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        return None

def _summarize_text_with_ai(text_content: str, filename: str) -> str | None:
    """
    使用 AI 模型对提取的文本进行总结。(此函数无需修改)
    """
    client = get_openai_client()
    if not client or not text_content.strip():
        return None

    prompt = f"""你是一个专业的证照信息识别助手。请根据以下从图片“{filename}”中识别出的文字，提取关键信息并以清晰的JSON格式返回。
你需要识别并提取的关键字段包括但不限于：'单位名称', '统一社会信用代码', '类型', '法定代表人', '注册资本', '成立日期', '营业期限', '住所'等。
如果某些字段不存在，请忽略。请确保JSON格式正确无误。

原始识别文字：
---
{text_content[:4000]}
---

请输出提取后的JSON对象：
"""
    logger.warning(f"Sending text from '{filename}' to AI for summarization...")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": settings.OPENROUTER_REFERER,
                "X-Title": settings.OPENROUTER_TITLE,
            },
            model=settings.OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        summary = completion.choices[0].message.content
        logger.warning(f"Successfully received summary for '{filename}'.")
        return summary
    except Exception as e:
        logger.error(f"Error during AI summarization for '{filename}': {e}", exc_info=True)
        return None

async def recognize_and_summarize_image(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    主处理函数：使用两阶段方法（检测+识别）处理图片，然后进行AI总结。
    """
    det_engine = get_detection_engine()
    rec_engine = get_recognition_engine()

    if not det_engine or not rec_engine:
        raise RuntimeError("OCR service is not available due to engine initialization failure.")

    image_np = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_cv is None:
        raise ValueError("无法解码图片，请确保文件格式正确。")

    # === STAGE 1: TEXT DETECTION (已修复) ===
    logger.warning(f"\n======== [开始] 1a. 文本位置检测 | 文件: {filename} ========")
    
    det_results_list = det_engine.predict(input=image_cv)
    
    if not det_results_list:
        logger.warning(f"文本检测未能为图片 '{filename}' 返回任何结果。")
        return {"ocr_results": [], "summary": None}
        
    det_result_obj = det_results_list[0]
    
    text_boxes = det_result_obj.get('dt_polys', np.array([]))

    if text_boxes.size == 0:
        logger.warning(f"在图片 '{filename}' 中未检测到任何文本框。")
        return {"ocr_results": [], "summary": None}
    logger.warning(f"======== [完成] 1a. 文本位置检测 | 检测到 {len(text_boxes)} 个文本框 ========")
    
    # === STAGE 2: TEXT RECOGNITION ===
    logger.warning(f"======== [开始] 1b. 文本内容识别 | 模型: PP-OCRv5_server_rec ========")
    cropped_images = [_crop_text_region(image_cv, np.array(box)) for box in text_boxes]
    rec_results = rec_engine.predict(input=cropped_images, batch_size=len(cropped_images))

    raw_text_parts = []
    formatted_results = []
    for i, res in enumerate(rec_results):
        box = [[int(p[0]), int(p[1])] for p in text_boxes[i]]
        
        # -------------------- 最终 Bug 修正点 (基于调试信息) --------------------
        # TextRecResult 也是一个类字典对象，我们用 .get() 来安全地访问
        text = res.get('rec_text', '')
        confidence = res.get('rec_score', 0.0)
        # -------------------- 修正结束 --------------------

        raw_text_parts.append(text)
        formatted_results.append({
            "box": box,
            "text": text,
            "confidence": confidence
        })

    full_text = "\n".join(raw_text_parts)
    logger.warning(f"======== [完成] 1b. 文本内容识别 | 提取总字符数: {len(full_text)} ========")

    # === STAGE 3: AI SUMMARY (逻辑不变) ===
    summary = None
    if full_text.strip():
        logger.warning(f"======== [开始] 2. AI 结构化总结 | 模型: {settings.OPENAI_MODEL_NAME} ========")
        summary = _summarize_text_with_ai(full_text, filename)
        if summary:
            logger.warning(f"======== [完成] 2. AI 结构化总结 ========")
        else:
            logger.warning(f"======== [失败] 2. AI 结构化总结未能返回结果 ========")
    else:
        logger.warning(f"OCR 未能从图片 '{filename}' 中提取任何文本，跳过 AI 总结步骤。")

    return {
        "ocr_results": formatted_results,
        "summary": summary
    }

# _crop_text_region 函数保持不变
def _crop_text_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    points = box.astype(np.float32)
    width = int(np.linalg.norm(points[0] - points[1]))
    height = int(np.linalg.norm(points[1] - points[2]))
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    cropped_image = cv2.warpPerspective(image, matrix, (width, height))
    return cropped_image