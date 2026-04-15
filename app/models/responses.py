from pydantic import BaseModel, Field


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
    use_ip_adapter: bool = False
    ip_adapter_image_path: str | None = None
    ip_adapter_scale: float = 0.6


class GenerationResponse(BaseModel):
    filename: str
    filenames: list[str] = Field(default_factory=list)
    prompt: str
    negative_prompt: str
    seed: int | None = None
    model_id: str | None = None
    original_prompt: str | None = None
    prompt_source_language: str | None = None
    prompt_was_translated: bool = False


class AnalyzeAndGenerateResponse(BaseModel):
    source_filename: str
    generated_filename: str
    generated_filenames: list[str] = Field(default_factory=list)
    subject: str
    hair: str
    clothing: str
    environment: str
    style: str
    prompt: str
    negative_prompt: str
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
    generated_filenames: list[str] = Field(default_factory=list)
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
    queue_position: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    original_prompt: str | None = None
    prompt_source_language: str | None = None
    prompt_was_translated: bool = False

class DiscoverModelResponse(BaseModel):
    id: str
    name: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    tags: list[str] = []
    installed: bool = False

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    translated: bool
    warning: str | None = None