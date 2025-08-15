# ocr_gateway/api/v1/recognize.py
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from ...services import ocr_service
from ...schemas.ocr import OCRResponse, RecognitionItem

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/recognize", response_model=OCRResponse)
async def create_upload_file(file: UploadFile = File(...)):
    """
    上传证照图片，进行 OCR 识别并返回 AI 总结。
    """
    # 检查 OCR 引擎是否已成功加载
    if ocr_service.get_recognition_engine() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service is currently unavailable. Check server logs for model loading errors."
        )

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="不支持的文件类型，请上传 JPG 或 PNG 图片。",
        )
    
    if not file.filename:
        file.filename = "untitled"

    try:
        contents = await file.read()
        
        service_result = await ocr_service.recognize_and_summarize_image(
            image_bytes=contents, 
            filename=str(file.filename)
        )
        
        return OCRResponse(
            filename=str(file.filename),
            data=[RecognitionItem(**item) for item in service_result["ocr_results"]],
            summary=service_result["summary"]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        # -------------------- 核心修改点 --------------------
        # 在返回 500 错误之前，在服务器控制台打印详细的错误堆栈
        logger.error(f"处理文件 '{file.filename}' 时发生未知内部错误", exc_info=True)
        # -------------------- 修改结束 --------------------
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"处理文件时发生未知内部错误: {e}"
        )