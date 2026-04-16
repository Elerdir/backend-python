from pydantic import BaseModel, Field, field_validator


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
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field("", max_length=2000)
    width: int = Field(1024, ge=64, le=2048)
    height: int = Field(1024, ge=64, le=2048)
    num_inference_steps: int = Field(30, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    seed: int | None = Field(None, ge=0)
    model_id: str | None = Field(None, max_length=200)
    num_images: int = Field(1, ge=1, le=16)
    use_ip_adapter: bool = False
    ip_adapter_image_path: str | None = None
    ip_adapter_scale: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("width", "height")
    @classmethod
    def must_be_multiple_of_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("width and height must be multiples of 8")
        return v


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