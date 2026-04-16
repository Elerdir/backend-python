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
    JobStartResponse,
    JobStatusResponse,
    DiscoverModelResponse,
)

from app.services.generation_service import GenerationService
from app.services.job_service import JobService
from app.dependencies import image_caption_service, caption_parser_service, prompt_builder_service, translation_service

router = APIRouter()

generation_service = GenerationService()
job_service = JobService()

# EXPORT shared instance
shared_generation_service = generation_service

UPLOAD_DIR = Path(os.getenv("APP_DATA_DIR", "data")) / "inputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class InstallModelRequest(BaseModel):
    model_id: str


class UninstallModelRequest(BaseModel):
    model_id: str


class UnloadModelRequest(BaseModel):
    model_id: str | None = None
    unload_all: bool = False


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


def join_unique(values: list[str | None]) -> str:
    cleaned = [v.strip() for v in values if v and v.strip()]
    unique = list(dict.fromkeys(cleaned))
    return ", ".join(unique) if unique else "unknown"


def prepare_prompt_for_generation(prompt: str) -> dict:
    result = translation_service.translate_to_english(prompt)

    return {
        "original_prompt": result.get("original_text", prompt),
        "final_prompt": result.get("translated_text", prompt),
        "prompt_source_language": result.get("source_language"),
        "prompt_was_translated": bool(result.get("translated", False)),
        "translation_warning": result.get("warning"),
    }


# --------------------------------------------------
# JOB RUNNERS
# --------------------------------------------------

def run_generate_job(job_id: str, job: dict) -> None:
    try:
        original_prompt = job.get("prompt", "")
        negative_prompt = job.get("negative_prompt", "")

        prompt_info = prepare_prompt_for_generation(original_prompt)
        prompt = prompt_info["final_prompt"]

        width = job.get("width", 512)
        height = job.get("height", 512)
        num_inference_steps = job.get("num_inference_steps", 12)
        guidance_scale = job.get("guidance_scale", 6.5)
        seed = job.get("seed")
        model_id = job.get("model_id")
        num_images = job.get("num_images", 1)

        def on_progress(progress: int, message: str) -> None:
            if job_service.is_cancelled(job_id):
                raise RuntimeError("Job cancelled")

            job_service.update_job(
                job_id,
                progress=progress,
                message=message,
            )

        if job_service.is_cancelled(job_id):
            raise RuntimeError("Job cancelled")

        job_service.update_job(
            job_id,
            progress=8,
            message="Preparing generation...",
            prompt=prompt,
            original_prompt=prompt_info["original_prompt"],
            prompt_source_language=prompt_info["prompt_source_language"],
            prompt_was_translated=prompt_info["prompt_was_translated"],
        )

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
            progress_callback=on_progress,
        )

        if job_service.is_cancelled(job_id):
            raise RuntimeError("Job cancelled")

        filenames = [Path(p).name for p in image_paths]

        job_service.update_job(
            job_id,
            progress=95,
            message="Finalizing generated images...",
            generated_filenames=filenames,
            prompt=prompt,
            original_prompt=prompt_info["original_prompt"],
            prompt_source_language=prompt_info["prompt_source_language"],
            prompt_was_translated=prompt_info["prompt_was_translated"],
            negative_prompt=negative_prompt,
            seed=seed,
            model_id=model_id,
        )

    except Exception as ex:
        if job_service.is_cancelled(job_id) or "cancelled" in str(ex).lower():
            job_service.update_job(
                job_id,
                status="cancelled",
                progress=100,
                message="Cancelled by user",
                error=None,
            )
        else:
            job_service.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Generation failed",
                error=str(ex),
            )
        raise


def run_analyze_and_generate_job(job_id: str, job: dict) -> None:
    try:
        input_file_paths: list[str] = job.get("input_file_paths", [])
        stored_filenames: list[str] = job.get("stored_filenames", [])
        original_filenames: list[str] = job.get("original_filenames", [])

        prompt_override = job.get("prompt_override", "")
        negative_prompt_override = job.get("negative_prompt_override", "")
        width = job.get("width", 512)
        height = job.get("height", 512)
        num_inference_steps = job.get("num_inference_steps", 12)
        guidance_scale = job.get("guidance_scale", 6.5)
        seed = job.get("seed")
        model_id = job.get("model_id")
        num_images = job.get("num_images", 1)

        if not input_file_paths:
            raise RuntimeError("No input files available for analysis")

        parsed_results = []

        job_service.update_job(
            job_id,
            progress=5,
            message="Preparing analysis...",
        )

        total_files = max(1, len(input_file_paths))

        for index, file_path_str in enumerate(input_file_paths):
            if job_service.is_cancelled(job_id):
                raise RuntimeError("Job cancelled")

            file_path = Path(file_path_str)

            caption = image_caption_service.analyze_image(str(file_path))
            parsed = caption_parser_service.parse_caption(caption)
            parsed_results.append(parsed)

            analysis_progress = 10 + int(((index + 1) / total_files) * 30)

            job_service.update_job(
                job_id,
                progress=analysis_progress,
                message=f"Analyzing reference images... {index + 1}/{total_files}",
            )

        if job_service.is_cancelled(job_id):
            raise RuntimeError("Job cancelled")

        analysis = AnalysisResponse(
            filename=stored_filenames[0] if stored_filenames else Path(input_file_paths[0]).name,
            original_filename=original_filenames[0] if original_filenames else Path(input_file_paths[0]).name,
            subject=join_unique([p.get("subject") for p in parsed_results]),
            hair=join_unique([p.get("hair") for p in parsed_results]),
            clothing=join_unique([p.get("clothing") for p in parsed_results]),
            environment=join_unique([p.get("environment") for p in parsed_results]),
            style=join_unique([p.get("style") for p in parsed_results]),
        )

        job_service.update_job(
            job_id,
            progress=45,
            message="Building prompt from analysis...",
            source_filename=analysis.filename,
            subject=analysis.subject,
            hair=analysis.hair,
            clothing=analysis.clothing,
            environment=analysis.environment,
            style=analysis.style,
        )

        prompt = prompt_builder_service.build_prompt(analysis)
        negative_prompt = prompt_builder_service.build_negative_prompt()

        final_prompt = prompt

        if prompt_override.strip():
            prompt_info = prepare_prompt_for_generation(prompt_override)
            final_prompt += f", {prompt_info['final_prompt']}"

            job_service.update_job(
                job_id,
                original_prompt=prompt_info["original_prompt"],
                prompt_source_language=prompt_info["prompt_source_language"],
                prompt_was_translated=prompt_info["prompt_was_translated"],
            )

        if negative_prompt_override.strip():
            negative_prompt += f", {negative_prompt_override.strip()}"

        job_service.update_job(
            job_id,
            progress=50,
            message="Starting image generation...",
            prompt=final_prompt,
            negative_prompt=negative_prompt,
        )

        def on_generation_progress(progress: int, message: str) -> None:
            if job_service.is_cancelled(job_id):
                raise RuntimeError("Job cancelled")

            mapped_progress = 50 + int(((progress - 10) / 85) * 45)
            mapped_progress = max(50, min(95, mapped_progress))

            job_service.update_job(
                job_id,
                progress=mapped_progress,
                message=message,
            )

        image_paths = generation_service.generate_images(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model_id=model_id,
            num_images=num_images,
            progress_callback=on_generation_progress,
        )

        if job_service.is_cancelled(job_id):
            raise RuntimeError("Job cancelled")

        filenames = [Path(p).name for p in image_paths]

        job_service.update_job(
            job_id,
            progress=95,
            message="Finalizing generated images...",
            generated_filenames=filenames,
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            model_id=model_id,
        )

    except Exception as ex:
        if job_service.is_cancelled(job_id) or "cancelled" in str(ex).lower():
            job_service.update_job(
                job_id,
                status="cancelled",
                progress=100,
                message="Cancelled by user",
                error=None,
            )
        else:
            job_service.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Analyze+Generate failed",
                error=str(ex),
            )
        raise


# --------------------------------------------------
# MODELS
# --------------------------------------------------

@router.get("/models")
async def list_models():
    return generation_service.list_models()


@router.post("/models/install")
async def install_model(request: InstallModelRequest):
    try:
        return generation_service.install_model(request.model_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Install error: {str(e)}"
        )


@router.post("/models/uninstall")
async def uninstall_model(request: UninstallModelRequest):
    try:
        return generation_service.uninstall_model(request.model_id)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Uninstall error: {str(e)}"
        )


@router.post("/models/unload")
async def unload_model(request: UnloadModelRequest):
    try:
        return generation_service.unload_model(
            model_id=request.model_id,
            unload_all=request.unload_all,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unload error: {str(e)}"
        )


@router.get("/models/discover", response_model=list[DiscoverModelResponse])
async def discover_models(
    search: str = "",
    limit: int = 20,
    task: str | None = None,
    only_diffusers: bool = False,
    only_sdxl: bool = False,
    installed_only: bool = False,
):
    try:
        return generation_service.discover_models(
            search=search,
            limit=limit,
            task=task,
            only_diffusers=only_diffusers,
            only_sdxl=only_sdxl,
            installed_only=installed_only,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Discover models error: {str(e)}"
        )


# --------------------------------------------------
# JOB GENERATE
# --------------------------------------------------

@router.post("/generate-job", response_model=JobStartResponse)
async def generate_job(request: GenerationRequest) -> JobStartResponse:
    generation_service.cleanup_old_files()
    job_service.cleanup_old_jobs()

    job_id = str(uuid.uuid4())

    job_service.create_job(
        job_id,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        model_id=request.model_id,
        source_filename=None,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        num_images=request.num_images or 1,
    )

    job_service.enqueue_job(job_id)
    job_service.start_worker(run_generate_job)

    return JobStartResponse(
        job_id=job_id,
        status="queued",
    )


@router.post("/analyze-and-generate-job", response_model=JobStartResponse)
async def analyze_and_generate_job(
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
) -> JobStartResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    generation_service.cleanup_old_files()
    job_service.cleanup_old_jobs()

    for file in files:
        validate_image_file(file.filename)

    stored_filenames: list[str] = []
    original_filenames: list[str] = []
    input_file_paths: list[str] = []

    try:
        for file in files:
            unique_name = f"{uuid.uuid4()}_{file.filename}"
            file_path = UPLOAD_DIR / unique_name

            contents = await file.read()

            with open(file_path, "wb") as f:
                f.write(contents)

            stored_filenames.append(unique_name)
            original_filenames.append(file.filename)
            input_file_paths.append(str(file_path))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store uploaded files: {str(e)}"
        )

    job_id = str(uuid.uuid4())

    job_service.create_job(
        job_id,
        prompt="",
        negative_prompt="",
        seed=seed,
        model_id=model_id,
        source_filename=stored_filenames[0] if stored_filenames else None,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images or 1,
    )

    job_service.update_job(
        job_id,
        input_file_paths=input_file_paths,
        stored_filenames=stored_filenames,
        original_filenames=original_filenames,
        prompt_override=prompt_override,
        negative_prompt_override=negative_prompt_override,
        message="Queued for analysis and generation",
    )

    job_service.enqueue_job(job_id)
    job_service.start_worker(run_analyze_and_generate_job)

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


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs() -> list[JobStatusResponse]:
    return [JobStatusResponse(**job) for job in job_service.list_jobs()]


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    success = job_service.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Job not found or not cancellable"
        )

    return {"status": "cancelling"}


@router.post("/generate-with-ip-adapter", response_model=GenerationResponse)
async def generate_with_ip_adapter(request: GenerationRequest):
    try:
        generation_service.cleanup_old_files()

        prompt_info = prepare_prompt_for_generation(request.prompt)
        final_prompt = prompt_info["final_prompt"]

        image_paths = generation_service.generate_images(
            prompt=final_prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            model_id=request.model_id,
            num_images=request.num_images or 1,
            use_ip_adapter=True,
            ip_adapter_image_path=request.ip_adapter_image_path,
            ip_adapter_scale=request.ip_adapter_scale,
        )

        filenames = [Path(p).name for p in image_paths]

        return GenerationResponse(
            filename=filenames[0],
            filenames=filenames,
            prompt=final_prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            model_id=request.model_id,
            original_prompt=prompt_info["original_prompt"],
            prompt_source_language=prompt_info["prompt_source_language"],
            prompt_was_translated=prompt_info["prompt_was_translated"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation with IP-Adapter error: {str(e)}"
        )

# --------------------------------------------------
# GENERATE (TEXT ONLY)
# --------------------------------------------------

@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    try:
        generation_service.cleanup_old_files()

        prompt_info = prepare_prompt_for_generation(request.prompt)
        final_prompt = prompt_info["final_prompt"]

        image_paths = generation_service.generate_images(
            prompt=final_prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            model_id=request.model_id,
            num_images=request.num_images or 1,
            use_ip_adapter=request.use_ip_adapter,
            ip_adapter_image_path=request.ip_adapter_image_path,
            ip_adapter_scale=request.ip_adapter_scale,
        )

        filenames = [Path(p).name for p in image_paths]

        return GenerationResponse(
            filename=filenames[0],
            filenames=filenames,
            prompt=final_prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            model_id=request.model_id,
            original_prompt=prompt_info["original_prompt"],
            prompt_source_language=prompt_info["prompt_source_language"],
            prompt_was_translated=prompt_info["prompt_was_translated"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


# --------------------------------------------------
# ANALYZE + GENERATE (SYNC, legacy compatibility)
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

    generation_service.cleanup_old_files()

    for file in files:
        validate_image_file(file.filename)

    stored_files = []
    parsed_results = []

    try:
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

        analysis = AnalysisResponse(
            filename=stored_files[0][0],
            original_filename=stored_files[0][2],
            subject=join_unique([p.get("subject") for p in parsed_results]),
            hair=join_unique([p.get("hair") for p in parsed_results]),
            clothing=join_unique([p.get("clothing") for p in parsed_results]),
            environment=join_unique([p.get("environment") for p in parsed_results]),
            style=join_unique([p.get("style") for p in parsed_results]),
        )

        prompt = prompt_builder_service.build_prompt(analysis)
        negative_prompt = prompt_builder_service.build_negative_prompt()

        if prompt_override.strip():
            prompt_info = prepare_prompt_for_generation(prompt_override)
            prompt += f", {prompt_info['final_prompt']}"

        if negative_prompt_override.strip():
            negative_prompt += f", {negative_prompt_override.strip()}"

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

        return AnalyzeAndGenerateResponse(
            source_filename=stored_files[0][0],
            generated_filename=filenames[0],
            generated_filenames=filenames,
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