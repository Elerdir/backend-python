from pathlib import Path
import uuid
import os
import threading

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.models.responses import (
    GenerationRequest,
    GenerationResponse,
    AnalysisResponse,
    AnalyzeAndGenerateResponse,
    JobStartResponse,
    JobStatusResponse,
)

from app.services.generation_service import GenerationService
from app.services.image_caption_service import ImageCaptionService
from app.services.caption_parser_service import CaptionParserService
from app.services.prompt_builder_service import PromptBuilderService
from app.services.job_service import JobService

router = APIRouter()

generation_service = GenerationService()
image_caption_service = ImageCaptionService()
caption_parser_service = CaptionParserService()
prompt_builder_service = PromptBuilderService()
job_service = JobService()

UPLOAD_DIR = Path(os.getenv("APP_DATA_DIR", "data")) / "inputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_image_file(filename: str) -> None:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}"
        )


def run_generate_job(
    job_id: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
    model_id: str | None,
    num_images: int,
) -> None:
    try:
        job_service.update_job(job_id, status="running", progress=10, message="Preparing generation...")

        image_paths = generation_service.generate_images(
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

        filenames = [Path(p).name for p in image_paths]

        job_service.update_job(
            job_id,
            status="completed",
            progress=100,
            message=f"Generated {len(filenames)} image(s)",
            generated_filenames=filenames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            model_id=model_id,
        )
    except Exception as ex:
        job_service.update_job(
            job_id,
            status="failed",
            progress=100,
            message="Generation failed",
            error=str(ex),
        )


# --------------------------------------------------
# MODELS
# --------------------------------------------------

@router.get("/models")
async def list_models():
    return generation_service.list_models()


@router.post("/models/install")
async def install_model(model_id: str):
    try:
        return generation_service.install_model(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Install error: {str(e)}"
        )


@router.post("/generate-job", response_model=JobStartResponse)
async def generate_job(request: GenerationRequest) -> JobStartResponse:
    generation_service.cleanup_old_files()

    job_id = str(uuid.uuid4())
    job_service.create_job(job_id)

    worker = threading.Thread(
        target=run_generate_job,
        args=(
            job_id,
            request.prompt,
            request.negative_prompt,
            request.width,
            request.height,
            request.num_inference_steps,
            request.guidance_scale,
            request.seed,
            request.model_id,
            request.num_images,
        ),
        daemon=True,
    )
    worker.start()

    return JobStartResponse(
        job_id=job_id,
        status="queued",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = job_service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(**job)

# --------------------------------------------------
# GENERATE (TEXT ONLY)
# --------------------------------------------------

@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    try:
        generation_service.cleanup_old_files()

        image_paths = generation_service.generate_images(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            model_id=request.model_id,
            num_images=request.num_images or 1,
        )

        filenames = [Path(p).name for p in image_paths]

        return GenerationResponse(
            filename=filenames[0],  # kompatibilita
            filenames=filenames,  # nové
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            model_id=request.model_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


# --------------------------------------------------
# ANALYZE + GENERATE
# --------------------------------------------------

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
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # 🔥 CLEANUP
    generation_service.cleanup_old_files()

    for file in files:
        validate_image_file(file.filename)

    stored_files = []
    parsed_results = []

    try:
        # --------------------------------------------------
        # SAVE + ANALYZE
        # --------------------------------------------------

        for file in files:
            unique_name = f"{uuid.uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / unique_name

            contents = await file.read()

            with open(file_path, "wb") as f:
                f.write(contents)

            stored_files.append((unique_name, file_path, file.filename))

            caption = image_caption_service.analyze_image(str(file_path))
            parsed = caption_parser_service.parse_caption(caption)

            parsed_results.append(parsed)

        # --------------------------------------------------
        # MERGE ANALYSIS
        # --------------------------------------------------

        def join_unique(values):
            cleaned = [v.strip() for v in values if v and v.strip()]
            unique = list(dict.fromkeys(cleaned))
            return ", ".join(unique) if unique else "unknown"

        analysis = AnalysisResponse(
            filename=stored_files[0][0],
            original_filename=stored_files[0][2],
            subject=join_unique([p.get("subject") for p in parsed_results]),
            hair=join_unique([p.get("hair") for p in parsed_results]),
            clothing=join_unique([p.get("clothing") for p in parsed_results]),
            environment=join_unique([p.get("environment") for p in parsed_results]),
            style=join_unique([p.get("style") for p in parsed_results]),
        )

        # --------------------------------------------------
        # PROMPT BUILD
        # --------------------------------------------------

        prompt = prompt_builder_service.build_prompt(analysis)
        negative_prompt = prompt_builder_service.build_negative_prompt()

        if prompt_override.strip():
            prompt += f", {prompt_override.strip()}"

        if negative_prompt_override.strip():
            negative_prompt += f", {negative_prompt_override.strip()}"

        # --------------------------------------------------
        # GENERATE
        # --------------------------------------------------

        image_paths = generation_service.generate_images(
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

        filenames = [Path(p).name for p in image_paths]

        # --------------------------------------------------
        # RESPONSE
        # --------------------------------------------------

        return AnalyzeAndGenerateResponse(
            source_filename=stored_files[0][0],
            generated_filename=filenames[0],  # kompatibilita
            generated_filenames=filenames,  # nové
            subject=analysis.subject,
            hair=analysis.hair,
            clothing=analysis.clothing,
            environment=analysis.environment,
            style=analysis.style,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            model_id=model_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analyze+Generate error: {str(e)}"
        )
