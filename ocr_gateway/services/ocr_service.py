# ocr_gateway/services/ocr_service.py
import logging
from typing import Dict, Any

import cv2
import numpy as np
from openai import OpenAI
from paddleocr import TextDetection, TextRecognition

from ..core.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 核心修改 1: 创建一个模型加载器函数 ---
def load_ocr_engines() -> Dict[str, Any]:
    """
    在应用启动时加载所有需要的 OCR 模型。
    返回一个包含所有引擎实例的字典。
    """
    logger.warning("--- 开始加载 OCR 模型 ---")
    
    # 加载文本检测模型
    detection_engine = TextDetection(device='gpu' if settings.OCR_USE_GPU else 'cpu')
    logger.warning("1/3: 文本检测模型加载成功。")

    # 加载默认的移动端识别模型
    recognition_mobile_engine = TextRecognition(
        model_name="PP-OCRv5_mobile_rec", # 速度更快，作为默认
        device='gpu' if settings.OCR_USE_GPU else 'cpu'
    )
    logger.warning("2/3: 移动端识别模型 (mobile) 加载成功。")

    # 加载高质量的服务器端识别模型
    recognition_server_engine = TextRecognition(
        model_name="PP-OCRv5_server_rec", # 精度更高
        device='gpu' if settings.OCR_USE_GPU else 'cpu'
    )
    logger.warning("3/3: 服务器端识别模型 (server) 加载成功。")

    logger.warning("--- 所有 OCR 模型加载完毕 ---")

    return {
        "detection": detection_engine,
        "recognition_mobile": recognition_mobile_engine,
        "recognition_server": recognition_server_engine
    }


def get_openai_client() -> OpenAI | None:
    """
    初始化并返回一个单例的 OpenAI 客户端。
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY 未在环境变量中设置。AI 总结功能将被禁用。")
        return None
    if not settings.OPENAI_BASE_URL:
        logger.warning("OPENAI_BASE_URL 未在环境变量中设置。AI 总结功能将被禁用。")
        return None

    try:
        logger.warning(f"正在初始化 OpenAI 客户端，目标模型: '{settings.OPENAI_MODEL_NAME}', API 地址: '{settings.OPENAI_BASE_URL}'")
        client = OpenAI(base_url=settings.OPENAI_BASE_URL, api_key=settings.OPENAI_API_KEY)
        return client
    except Exception as e:
        logger.error(f"初始化 OpenAI 客户端失败: {e}", exc_info=True)
        return None

# AI 总结函数保持不变
def _summarize_text_with_ai(text_content: str, filename: str) -> str | None:
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


# --- 核心修改 3: 修改核心函数签名，接收引擎作为参数 ---
async def recognize_and_summarize_image(
    *, # 强制使用关键字参数，增加代码可读性
    det_engine: TextDetection,
    rec_engine: TextRecognition,
    image_bytes: bytes,
    filename: str
) -> Dict[str, Any]:
    """
    主处理函数：使用传入的引擎处理图片，然后进行AI总结。
    """
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_cv is None:
        raise ValueError("无法解码图片，请确保文件格式正确。")

    # === STAGE 1: TEXT DETECTION (使用传入的引擎) ===
    logger.warning(f"\n======== [开始] 1a. 文本位置检测 | 文件: {filename} ========")
    det_results_list = det_engine.predict(input=image_cv)
    if not det_results_list:
        return {"ocr_results": [], "summary": None}
    text_boxes = det_results_list[0].get('dt_polys', np.array([]))
    if text_boxes.size == 0:
        logger.warning(f"在图片 '{filename}' 中未检测到任何文本框。")
        return {"ocr_results": [], "summary": None}
    logger.warning(f"======== [完成] 1a. 文本位置检测 | 检测到 {len(text_boxes)} 个文本框 ========")
    
    # === STAGE 2: TEXT RECOGNITION (使用传入的引擎) ===
    logger.warning(f"======== [开始] 1b. 文本内容识别 | 模型: {rec_engine._model_name} ========")
    cropped_images = [_crop_text_region(image_cv, np.array(box)) for box in text_boxes]
    rec_results = rec_engine.predict(input=cropped_images, batch_size=len(cropped_images))

    raw_text_parts = []
    formatted_results = []
    for i, res in enumerate(rec_results):
        box = [[int(p[0]), int(p[1])] for p in text_boxes[i]]
        text = res.get('rec_text', '')
        confidence = res.get('rec_score', 0.0)
        raw_text_parts.append(text)
        formatted_results.append({"box": box, "text": text, "confidence": confidence})

    full_text = "\n".join(raw_text_parts)
    logger.warning(f"======== [完成] 1b. 文本内容识别 | 提取总字符数: {len(full_text)} ========")

    # === STAGE 3: AI SUMMARY (逻辑不变) ===
    summary = None
    if full_text.strip():
        # ... (AI总结部分代码与之前版本完全相同，此处省略) ...
        logger.warning(f"======== [开始] 2. AI 结构化总结 | 模型: {settings.OPENAI_MODEL_NAME} ========")
        summary = _summarize_text_with_ai(full_text, filename)
        if summary:
            logger.warning(f"======== [完成] 2. AI 结构化总结 ========")
        else:
            logger.warning(f"======== [失败] 2. AI 结构化总结未能返回结果 ========")

    return { "ocr_results": formatted_results, "summary": summary }

# _crop_text_region 函数保持不变
def _crop_text_region(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    # ... (代码与之前版本完全相同，此处省略) ...
    points = box.astype(np.float32)
    width = int(np.linalg.norm(points[0] - points[1]))
    height = int(np.linalg.norm(points[1] - points[2]))
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    cropped_image = cv2.warpPerspective(image, matrix, (width, height))
    return cropped_image