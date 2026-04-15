from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.responses import TranslationResponse
from app.services.translation_service import TranslationService

router = APIRouter()

translation_service = TranslationService()


class TranslationRequest(BaseModel):
    text: str


@router.post("/translate-to-english", response_model=TranslationResponse)
async def translate_to_english(request: TranslationRequest) -> TranslationResponse:
    try:
        result = translation_service.translate_to_english(request.text)
        return TranslationResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )