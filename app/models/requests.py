from pydantic import BaseModel

class GenerationResponse(BaseModel):
    filename: str
    prompt: str
    negative_prompt: str
    image_path: str

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None

class AnalyzeAndGenerateOptions(BaseModel):
    prompt_override: str = ""
    negative_prompt_override: str = ""