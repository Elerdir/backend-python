from pathlib import Path
import uuid
import json
import os

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


class GenerationService:
    def __init__(self) -> None:
        # directories
        self.base_dir = Path(os.getenv("APP_DATA_DIR", "data"))
        self.output_dir = self.base_dir / "outputs"
        self.models_dir = self.base_dir / "models"
        self.registry_path = self.base_dir / "models.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # default model
        self.default_model_id = "runwayml/stable-diffusion-v1-5"

        # pipeline cache
        self._pipelines: dict[str, object] = {}

        # ensure default registry
        self._ensure_default_registry()

        # preload default pipeline
        self.pipe = self._load_pipeline(self.default_model_id)

    # --------------------------------------------------
    # REGISTRY
    # --------------------------------------------------

    def _load_registry_safe(self) -> list[dict]:
        if not self.registry_path.exists():
            return []

        text = self.registry_path.read_text(encoding="utf-8").strip()

        if not text:
            return []

        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_registry(self, registry: list[dict]) -> None:
        self.registry_path.write_text(
            json.dumps(registry, indent=2),
            encoding="utf-8"
        )

    def _ensure_default_registry(self) -> None:
        registry = self._load_registry_safe()

        if not any(m.get("id") == self.default_model_id for m in registry):
            registry.insert(0, {
                "id": self.default_model_id,
                "name": "Stable Diffusion 1.5",
                "installed": True,
                "path": None,
                "pipeline_type": "sd"
            })
            self._save_registry(registry)

    def _get_model_info(self, model_id: str) -> dict | None:
        registry = self._load_registry_safe()

        for item in registry:
            if item.get("id") == model_id:
                return item

        return None

    # --------------------------------------------------
    # MODEL TYPE
    # --------------------------------------------------

    def _detect_pipeline_type(self, model_id: str) -> str:
        lower = model_id.lower()

        if "sdxl" in lower or "xl" in lower:
            return "sdxl"

        return "sd"

    def _resolve_model_path(self, model_id: str) -> Path:
        return self.models_dir / model_id.replace("/", "__")

    # --------------------------------------------------
    # PIPELINE LOAD
    # --------------------------------------------------

    def _load_pipeline(self, model_id: str):
        if model_id in self._pipelines:
            return self._pipelines[model_id]

        model_info = self._get_model_info(model_id)
        pipeline_type = model_info.get("pipeline_type") if model_info else self._detect_pipeline_type(model_id)

        model_path = self._resolve_model_path(model_id)
        source = str(model_path) if model_path.exists() else model_id

        print(f"[MODEL] Loading: {model_id} | type={pipeline_type} | source={source}")

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if pipeline_type == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                source,
                torch_dtype=torch_dtype,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                source,
                torch_dtype=torch_dtype,
            )

        pipe = pipe.to(self.device)

        if self.device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        self._pipelines[model_id] = pipe
        return pipe

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        model_id: str | None = None,
        num_images: int = 1,
    ) -> list[str]:
        selected_model_id = model_id or self.default_model_id
        self.pipe = self._load_pipeline(selected_model_id)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=max(1, num_images),
        )

        saved_paths: list[str] = []

        for image in result.images:
            filename = f"{uuid.uuid4()}.png"
            output_path = self.output_dir / filename
            image.save(output_path)
            saved_paths.append(str(output_path))

        return saved_paths

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        model_id: str | None = None,
        num_images: int = 1,
    ) -> str:
        return self.generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model_id=model_id,
            num_images=num_images,
        )[0]

    # --------------------------------------------------
    # MODELS LIST
    # --------------------------------------------------

    def list_models(self) -> list[dict]:
        registry = self._load_registry_safe()

        models: list[dict] = []

        for item in registry:
            models.append({
                "id": item.get("id", ""),
                "name": item.get("name", item.get("id", "")),
                "installed": bool(item.get("installed", True)),
            })

        return models

    # --------------------------------------------------
    # INSTALL MODEL
    # --------------------------------------------------

    def install_model(self, model_id: str) -> dict:
        print(f"[MODEL] Installing: {model_id}")

        model_path = self._resolve_model_path(model_id)

        if model_path.exists():
            registry = self._load_registry_safe()

            if not any(m.get("id") == model_id for m in registry):
                registry.append({
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "installed": True,
                    "path": str(model_path),
                    "pipeline_type": self._detect_pipeline_type(model_id)
                })
                self._save_registry(registry)

            return {"status": "already_installed"}

        pipeline_type = self._detect_pipeline_type(model_id)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            if pipeline_type == "sdxl":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                )

            pipe.save_pretrained(model_path)

            registry = self._load_registry_safe()

            if not any(m.get("id") == model_id for m in registry):
                registry.append({
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "installed": True,
                    "path": str(model_path),
                    "pipeline_type": pipeline_type
                })
                self._save_registry(registry)

            if model_id in self._pipelines:
                del self._pipelines[model_id]

            print(f"[MODEL] Installed: {model_id} | type={pipeline_type}")
            return {"status": "installed", "pipeline_type": pipeline_type}

        except Exception as ex:
            print(f"[MODEL] Install failed: {model_id} | {ex}")
            raise