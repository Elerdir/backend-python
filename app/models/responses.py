from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    filename: str
    original_filename: str
    subject: str
    hair: str
    clothing: str
    environment: str
    style: str


class ImageCaptionResponse(BaseModel):
    filename: str
    caption: str


class PromptResponse(BaseModel):
    filename: str
    original_filename: str
    subject: str
    hair: str
    clothing: str
    environment: str
    style: str
    prompt: str
    negative_prompt: str


from pydantic import BaseModel


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None
    model_id: str | None = None
    num_images: int = 1


class GenerationResponse(BaseModel):
    filename: str
    generated_filenames: list[str] = []
    prompt: str
    negative_prompt: str
    image_path: str
    image_paths: list[str] = []
    seed: int | None = None
    model_id: str | None = None


class AnalyzeAndGenerateResponse(BaseModel):
    source_filename: str
    generated_filename: str
    generated_filenames: list[str] = []
    subject: str
    hair: str
    clothing: str
    environment: str
    style: str
    prompt: str
    negative_prompt: str
    image_path: str
    image_paths: list[str] = []
    seed: int | None = None
    model_id: str | None = None

class JobStartResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    message: str = ""
    generated_filenames: list[str] = []
    source_filename: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    subject: str = ""
    hair: str = ""
    clothing: str = ""
    environment: str = ""
    style: str = ""
    seed: int | None = None
    model_id: str | None = None
    error: str | None = None