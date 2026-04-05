from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import uuid

from app.models.responses import AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.image_caption_service import ImageCaptionService
from app.services.caption_parser_service import CaptionParserService
from app.models.responses import AnalysisResponse, PromptResponse
from app.services.prompt_builder_service import PromptBuilderService

router = APIRouter()

analysis_service = AnalysisService()
image_caption_service = ImageCaptionService()
caption_parser_service = CaptionParserService()
prompt_builder_service = PromptBuilderService()

UPLOAD_DIR = Path("data/inputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_file(filename: str) -> None:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed types are: .jpg, .jpeg, .png"
        )


@router.post("/analyze-image", response_model=PromptResponse)
async def analyze_image(file: UploadFile = File(...)) -> PromptResponse:
    validate_image_file(file.filename)

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / unique_name

    try:
        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        caption = image_caption_service.analyze_image(str(file_path))
        print(f"Caption: {caption}")

        parsed = caption_parser_service.parse_caption(caption)
        print(f"Parsed: {parsed}")

        analysis = AnalysisResponse(
            filename=unique_name,
            original_filename=file.filename,
            subject=parsed.get("subject", "unknown"),
            hair=parsed.get("hair", "unknown"),
            clothing=parsed.get("clothing", "unknown"),
            environment=parsed.get("environment", "unknown"),
            style=parsed.get("style", "unknown"),
        )

        prompt = prompt_builder_service.build_prompt(analysis)
        negative_prompt = prompt_builder_service.build_negative_prompt()

        return PromptResponse(
            filename=analysis.filename,
            original_filename=analysis.original_filename,
            subject=analysis.subject,
            hair=analysis.hair,
            clothing=analysis.clothing,
            environment=analysis.environment,
            style=analysis.style,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file {file.filename}: {str(e)}"
        )


@router.post("/analyze-images", response_model=List[AnalysisResponse])
async def analyze_images(files: List[UploadFile] = File(...)) -> List[AnalysisResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results: List[AnalysisResponse] = []

    for file in files:
        validate_image_file(file.filename)

        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOAD_DIR / unique_name

        try:
            contents = await file.read()

            with open(file_path, "wb") as f:
                f.write(contents)

            result = analysis_service.analyze_mock(unique_name, file.filename)
            results.append(result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file {file.filename}: {str(e)}"
            )

    return results