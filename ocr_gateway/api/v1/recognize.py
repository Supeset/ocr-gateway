# ocr_gateway/api/v1/recognize.py
import logging
from enum import Enum
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Request, Query
from ...services import ocr_service
from ...schemas.ocr import OCRResponse, RecognitionItem

router = APIRouter()
logger = logging.getLogger(__name__)

# --- 核心修改 1: 定义一个枚举类来限制模型选择 ---
class ModelName(str, Enum):
    mobile = "mobile" # 默认模型，速度快
    server = "server" # 高质量模型，精度高

@router.post("/recognize", response_model=OCRResponse)
async def create_upload_file(
    # --- 核心修改 2: 注入 Request 和添加 Query 参数 ---
    request: Request,
    file: UploadFile = File(...),
    model: ModelName = Query(
        ModelName.mobile, # 默认值为 'mobile'
        description="选择要使用的识别模型：'mobile' (速度快) 或 'server' (精度高)。"
    )
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="不支持的文件类型。")
    
    try:
        # --- 核心修改 3: 从 app.state 获取预加载的模型 ---
        det_engine = request.app.state.ocr_engines["detection"]
        # 根据查询参数动态选择识别引擎
        rec_engine = request.app.state.ocr_engines[f"recognition_{model.value}"]

        contents = await file.read()
        
        # 将选择好的引擎传递给服务函数
        service_result = await ocr_service.recognize_and_summarize_image(
            det_engine=det_engine,
            rec_engine=rec_engine,
            image_bytes=contents, 
            filename=str(file.filename)
        )
        
        return OCRResponse(
            filename=str(file.filename),
            data=[RecognitionItem(**item) for item in service_result["ocr_results"]],
            summary=service_result["summary"]
        )
    except Exception as e:
        logger.error(f"处理文件 '{file.filename}' 时发生未知内部错误", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理文件时发生未知内部错误: {e}")