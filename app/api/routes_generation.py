from pathlib import Path
import uuid
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.models.responses import (
    GenerationRequest,
    GenerationResponse,
    AnalysisResponse,
    AnalyzeAndGenerateResponse,
)
from app.services.generation_service import GenerationService
from app.services.image_caption_service import ImageCaptionService
from app.services.caption_parser_service import CaptionParserService
from app.services.prompt_builder_service import PromptBuilderService

router = APIRouter()

generation_service = GenerationService()
image_caption_service = ImageCaptionService()
caption_parser_service = CaptionParserService()
prompt_builder_service = PromptBuilderService()

UPLOAD_DIR = Path(os.getenv("APP_DATA_DIR", "data")) / "inputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class InstallModelRequest(BaseModel):
    model_id: str


def validate_image_file(filename: str) -> None:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed types are: .jpg, .jpeg, .png"
        )


@router.post("/models/install")
async def install_model(request: InstallModelRequest):
    try:
        result = generation_service.install_model(request.model_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Install error: {str(e)}"
        )


@router.get("/models")
async def list_models() -> list[dict]:
    return generation_service.list_models()


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest) -> GenerationResponse:
    try:
        image_paths = generation_service.generate_images(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            model_id=request.model_id,
            num_images=request.num_images,
        )

        generated_filenames = [Path(p).name for p in image_paths]

        return GenerationResponse(
            filename=generated_filenames[0],
            generated_filenames=generated_filenames,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image_path=image_paths[0],
            image_paths=image_paths,
            seed=request.seed,
            model_id=request.model_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


@router.post("/analyze-and-generate", response_model=AnalyzeAndGenerateResponse)
async def analyze_and_generate(
    files: list[UploadFile] = File(...),
    prompt_override: str = Form(""),
    negative_prompt_override: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    num_inference_steps: int = Form(12),
    guidance_scale: float = Form(6.5),
    seed: int | None = Form(None),
    model_id: str | None = Form(None),
    num_images: int = Form(1),
) -> AnalyzeAndGenerateResponse:
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded."
        )

    for file in files:
        validate_image_file(file.filename)

    stored_files: list[tuple[str, Path, str]] = []
    parsed_results: list[dict] = []

    try:
        for file in files:
            unique_name = f"{uuid.uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / unique_name

            contents = await file.read()

            with open(file_path, "wb") as f:
                f.write(contents)

            stored_files.append((unique_name, file_path, file.filename))

            caption = image_caption_service.analyze_image(str(file_path))
            print(f"Caption for {file.filename}: {caption}")

            parsed = caption_parser_service.parse_caption(caption)
            print(f"Parsed for {file.filename}: {parsed}")

            parsed_results.append(parsed)

        subjects = [p.get("subject", "unknown") for p in parsed_results]
        hairs = [p.get("hair", "unknown") for p in parsed_results]
        clothings = [p.get("clothing", "unknown") for p in parsed_results]
        environments = [p.get("environment", "unknown") for p in parsed_results]
        styles = [p.get("style", "unknown") for p in parsed_results]

        def join_unique(values: list[str]) -> str:
            cleaned = [v.strip() for v in values if v and v.strip()]
            unique = list(dict.fromkeys(cleaned))
            return ", ".join(unique) if unique else "unknown"

        analysis = AnalysisResponse(
            filename=stored_files[0][0],
            original_filename=stored_files[0][2],
            subject=join_unique(subjects),
            hair=join_unique(hairs),
            clothing=join_unique(clothings),
            environment=join_unique(environments),
            style=join_unique(styles),
        )

        prompt = prompt_builder_service.build_prompt(analysis)
        negative_prompt = prompt_builder_service.build_negative_prompt()

        if prompt_override.strip():
            prompt = f"{prompt}, {prompt_override.strip()}"

        if negative_prompt_override.strip():
            negative_prompt = f"{negative_prompt}, {negative_prompt_override.strip()}"

        generated_image_paths = generation_service.generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model_id=model_id,
            num_images=num_images,
        )

        generated_filenames = [Path(p).name for p in generated_image_paths]

        return AnalyzeAndGenerateResponse(
            source_filename=stored_files[0][0],
            generated_filename=generated_filenames[0],
            generated_filenames=generated_filenames,
            subject=analysis.subject,
            hair=analysis.hair,
            clothing=analysis.clothing,
            environment=analysis.environment,
            style=analysis.style,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=generated_image_paths[0],
            image_paths=generated_image_paths,
            seed=seed,
            model_id=model_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analyze and generate error: {str(e)}"
        )