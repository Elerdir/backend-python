from pathlib import Path
from datetime import datetime, timedelta
import uuid
import json
import os
import inspect
import time
import gc
from typing import Callable, Any
import shutil
from huggingface_hub import HfApi
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


ProgressCallback = Callable[[int, str], None]


class GenerationService:
    def __init__(self) -> None:
        # directories
        self.base_dir = Path(os.getenv("APP_DATA_DIR", "data"))
        self.output_dir = self.base_dir / "outputs"
        self.inputs_dir = self.base_dir / "inputs"
        self.models_dir = self.base_dir / "models"
        self.registry_path = self.base_dir / "models.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._ip_adapter_loaded_for: set[str] = set()

        self.ip_adapter_repo = "h94/IP-Adapter"
        self.ip_adapter_subfolder = "sdxl_models"
        self.ip_adapter_weight_name = "ip-adapter_sdxl.bin"

        self.hf_api = HfApi()

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # default model
        self.default_model_id = "runwayml/stable-diffusion-v1-5"

        self.server_version = os.getenv("APP_VERSION", "1.0.0")
        self.app_name = "Image Studio Backend"

        # pipeline cache
        self._pipelines: dict[str, Any] = {}
        self.pipe: Any | None = None

        self._log_runtime_info()

        # ensure registry contains default model
        self._ensure_default_registry()

        # cleanup old temp/generated files
        self.cleanup_old_files()

        # preload default pipeline
        try:
            preload_default = os.getenv("PRELOAD_DEFAULT_MODEL", "true").lower() in ("1", "true", "yes")
            if preload_default:
                print(f"[MODEL] Preloading default model: {self.default_model_id}")
                self.pipe = self._load_pipeline(self.default_model_id)
        except Exception as ex:
            print(f"[MODEL] Default preload failed: {ex}")

    def _is_sdxl_model(self, model_id: str) -> bool:
        model_info = self._get_model_info(model_id)
        pipeline_type = model_info.get("pipeline_type") if model_info else self._detect_pipeline_type(model_id)
        return pipeline_type == "sdxl"

    def _ensure_ip_adapter_loaded(self, model_id: str) -> None:
        if self.pipe is None:
            raise RuntimeError("Pipeline is not loaded")

        if model_id in self._ip_adapter_loaded_for:
            return

        if not self._is_sdxl_model(model_id):
            raise RuntimeError("IP-Adapter is currently supported only for SDXL models")

        print(f"[IPADAPTER] Loading IP-Adapter for model: {model_id}")

        self.pipe.load_ip_adapter(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_subfolder,
            weight_name=self.ip_adapter_weight_name,
        )

        self._ip_adapter_loaded_for.add(model_id)
        print(f"[IPADAPTER] IP-Adapter loaded for model: {model_id}")

    def _load_ip_adapter_image(self, image_path: str) -> Image.Image:
        if not image_path or not image_path.strip():
            raise RuntimeError("ip_adapter_image_path is empty")

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"IP-Adapter image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        return image

    # --------------------------------------------------
    # RUNTIME / DIAGNOSTICS
    # --------------------------------------------------

    def _log_runtime_info(self) -> None:
        print(f"[TORCH] torch version: {torch.__version__}")
        print(f"[TORCH] cuda available: {torch.cuda.is_available()}")
        print(f"[TORCH] selected device: {self.device}")

        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                print(f"[TORCH] cuda device count: {torch.cuda.device_count()}")
                print(f"[TORCH] current cuda device: {current_device}")
                print(f"[TORCH] cuda device name: {torch.cuda.get_device_name(current_device)}")
                print(f"[TORCH] cuda capability: {torch.cuda.get_device_capability(current_device)}")
            except Exception as ex:
                print(f"[TORCH] Failed to read CUDA device info: {ex}")
        else:
            print("[TORCH] CUDA is not available. Generation will run on CPU.")

    def get_runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "app_name": self.app_name,
            "server_version": self.server_version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "selected_device": self.device,
            "default_model_id": self.default_model_id,
            "cached_models": list(self._pipelines.keys()),
        }

        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                info.update({
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_current_device": current_device,
                    "cuda_device_name": torch.cuda.get_device_name(current_device),
                    "cuda_device_capability": torch.cuda.get_device_capability(current_device),
                })

                try:
                    free_mem, total_mem = torch.cuda.mem_get_info(current_device)
                    info.update({
                        "cuda_memory_free_mb": round(free_mem / 1024 / 1024, 2),
                        "cuda_memory_total_mb": round(total_mem / 1024 / 1024, 2),
                    })
                except Exception:
                    pass
            except Exception as ex:
                info["cuda_info_error"] = str(ex)

        return info

    def _get_torch_dtype(self):
        return torch.float16 if self.device == "cuda" else torch.float32

    def _empty_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f"[CUDA] Failed to empty CUDA cache: {ex}")

    def unload_all_pipelines(self) -> None:
        print("[MODEL] Unloading all cached pipelines")
        self._pipelines.clear()
        self.pipe = None
        gc.collect()
        self._empty_cuda_cache()

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        directories = [
            self.inputs_dir,
            self.output_dir,
        ]

        for directory in directories:
            if not directory.exists():
                continue

            for file_path in directory.rglob("*"):
                if not file_path.is_file():
                    continue

                try:
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if modified < cutoff:
                        file_path.unlink(missing_ok=True)
                        print(f"[CLEANUP] Deleted old file: {file_path}")
                except Exception as ex:
                    print(f"[CLEANUP] Failed to delete {file_path}: {ex}")

        for directory in directories:
            if not directory.exists():
                continue

            for subdir in sorted(directory.rglob("*"), reverse=True):
                if not subdir.is_dir():
                    continue

                try:
                    if not any(subdir.iterdir()):
                        subdir.rmdir()
                        print(f"[CLEANUP] Deleted empty directory: {subdir}")
                except Exception:
                    pass

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
        # Write to a temp file first, then atomically replace the target.
        # This prevents registry corruption if the process crashes mid-write.
        tmp_path = self.registry_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(registry, indent=2),
                encoding="utf-8"
            )
            tmp_path.replace(self.registry_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

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
            print(f"[MODEL] Using cached pipeline: {model_id}")
            return self._pipelines[model_id]

        model_info = self._get_model_info(model_id)
        pipeline_type = model_info.get("pipeline_type") if model_info else self._detect_pipeline_type(model_id)

        model_path = self._resolve_model_path(model_id)
        source = str(model_path) if model_path.exists() else model_id

        torch_dtype = self._get_torch_dtype()

        print(f"[MODEL] Loading: {model_id} | type={pipeline_type} | source={source}")
        print(f"[MODEL] torch_dtype: {torch_dtype}")
        print(f"[MODEL] target device: {self.device}")

        start_load = time.perf_counter()

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

        print(f"[MODEL] Pipeline instantiated in {time.perf_counter() - start_load:.2f}s")

        start_to_device = time.perf_counter()
        pipe = pipe.to(self.device)
        print(f"[MODEL] Pipeline moved to {self.device} in {time.perf_counter() - start_to_device:.2f}s")

        if self.device == "cuda":
            try:
                pipe.enable_attention_slicing()
                print("[MODEL] attention slicing enabled")
            except Exception as ex:
                print(f"[MODEL] attention slicing not enabled: {ex}")

            try:
                # Optional and environment-dependent
                if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    pipe.enable_xformers_memory_efficient_attention()
                    print("[MODEL] xformers memory efficient attention enabled")
            except Exception as ex:
                print(f"[MODEL] xformers not enabled: {ex}")

        self._pipelines[model_id] = pipe
        print(f"[MODEL] Pipeline ready and cached: {model_id}")

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
            progress_callback: ProgressCallback | None = None,
            use_ip_adapter: bool = False,
            ip_adapter_image_path: str | None = None,
            ip_adapter_scale: float = 0.6,
    ) -> list[str]:
        total_start = time.perf_counter()

        self.cleanup_old_files()

        selected_model_id = model_id or self.default_model_id

        print(
            f"[GEN] Starting generation | model={selected_model_id} | "
            f"device={self.device} | size={width}x{height} | "
            f"steps={num_inference_steps} | cfg={guidance_scale} | "
            f"images={num_images} | seed={seed}"
        )

        if progress_callback:
            progress_callback(10, "Loading model...")

        load_start = time.perf_counter()
        self.pipe = self._load_pipeline(selected_model_id)
        print(f"[PERF] pipeline load total: {time.perf_counter() - load_start:.2f}s")

        ip_adapter_image = None

        if use_ip_adapter:
            if not ip_adapter_image_path:
                raise RuntimeError("use_ip_adapter=True but no ip_adapter_image_path was provided")

            if progress_callback:
                progress_callback(15, "Loading IP-Adapter...")

            self._ensure_ip_adapter_loaded(selected_model_id)
            self.pipe.set_ip_adapter_scale(ip_adapter_scale)
            ip_adapter_image = self._load_ip_adapter_image(ip_adapter_image_path)

            print(
                f"[IPADAPTER] enabled | model={selected_model_id} | "
                f"scale={ip_adapter_scale} | image={ip_adapter_image_path}"
            )

        if progress_callback:
            progress_callback(20, "Running inference...")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        pipe_call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": max(1, num_images),
        }

        if ip_adapter_image is not None:
            pipe_call_kwargs["ip_adapter_image"] = ip_adapter_image

        def emit_step_progress(step_index: int) -> None:
            if not progress_callback:
                return

            total_steps = max(1, num_inference_steps)
            ratio = min(1.0, max(0.0, (step_index + 1) / total_steps))
            progress = 20 + int(ratio * 70)  # 20..90
            progress_callback(progress, f"Generating... step {step_index + 1}/{total_steps}")

        call_signature = inspect.signature(self.pipe.__call__)

        if "callback_on_step_end" in call_signature.parameters:
            def on_step_end(pipe, step_index, timestep, callback_kwargs):
                emit_step_progress(step_index)
                return callback_kwargs

            pipe_call_kwargs["callback_on_step_end"] = on_step_end

        elif "callback" in call_signature.parameters:
            def legacy_callback(step: int, timestep: int, latents) -> None:
                emit_step_progress(step)

            pipe_call_kwargs["callback"] = legacy_callback

            if "callback_steps" in call_signature.parameters:
                pipe_call_kwargs["callback_steps"] = 1

        inference_start = time.perf_counter()

        try:
            result = self.pipe(**pipe_call_kwargs)
        except torch.cuda.OutOfMemoryError as ex:
            print(f"[CUDA] Out of memory during generation: {ex}")
            self._empty_cuda_cache()
            raise RuntimeError(
                "CUDA out of memory. Try lower resolution, fewer images, fewer steps, or unload other models."
            ) from ex

        print(f"[PERF] inference: {time.perf_counter() - inference_start:.2f}s")

        if progress_callback:
            progress_callback(95, "Saving generated images...")

        save_start = time.perf_counter()

        saved_paths: list[str] = []

        for image in result.images:
            filename = f"{uuid.uuid4()}.png"
            output_path = self.output_dir / filename
            image.save(output_path)
            saved_paths.append(str(output_path))

        print(f"[PERF] saving images: {time.perf_counter() - save_start:.2f}s")
        print(f"[PERF] total generation: {time.perf_counter() - total_start:.2f}s")

        if self.device == "cuda":
            try:
                current_device = torch.cuda.current_device()
                allocated_mb = torch.cuda.memory_allocated(current_device) / 1024 / 1024
                reserved_mb = torch.cuda.memory_reserved(current_device) / 1024 / 1024
                print(f"[CUDA] memory allocated: {allocated_mb:.2f} MB | reserved: {reserved_mb:.2f} MB")
            except Exception as ex:
                print(f"[CUDA] Failed to read memory stats: {ex}")

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
            progress_callback: ProgressCallback | None = None,
            use_ip_adapter: bool = False,
            ip_adapter_image_path: str | None = None,
            ip_adapter_scale: float = 0.6,
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
            progress_callback=progress_callback,
            use_ip_adapter=use_ip_adapter,
            ip_adapter_image_path=ip_adapter_image_path,
            ip_adapter_scale=ip_adapter_scale,
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
        torch_dtype = self._get_torch_dtype()

        install_start = time.perf_counter()

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

            del pipe
            gc.collect()
            self._empty_cuda_cache()

            print(f"[MODEL] Installed: {model_id} | type={pipeline_type}")
            print(f"[PERF] model install: {time.perf_counter() - install_start:.2f}s")

            return {"status": "installed", "pipeline_type": pipeline_type}

        except Exception as ex:
            print(f"[MODEL] Install failed: {model_id} | {ex}")
            raise

    def uninstall_model(self, model_id: str) -> dict:
        print(f"[MODEL] Uninstalling: {model_id}")

        if not model_id or not model_id.strip():
            raise ValueError("model_id is required")

        model_id = model_id.strip()

        if model_id == self.default_model_id:
            raise ValueError("Default model cannot be uninstalled")

        model_info = self._get_model_info(model_id)
        model_path = self._resolve_model_path(model_id)

        # Remove cached pipeline if loaded
        if model_id in self._pipelines:
            try:
                del self._pipelines[model_id]
            except Exception:
                pass

        if self.pipe is not None:
            try:
                current_pipe = self._pipelines.get(model_id)
                if current_pipe is not None and self.pipe is current_pipe:
                    self.pipe = None
            except Exception:
                pass

        gc.collect()
        self._empty_cuda_cache()

        removed_from_disk = False
        if model_path.exists() and model_path.is_dir():
            shutil.rmtree(model_path, ignore_errors=True)
            removed_from_disk = True

        registry = self._load_registry_safe()
        new_registry = [m for m in registry if m.get("id") != model_id]

        removed_from_registry = len(new_registry) != len(registry)

        if removed_from_registry:
            self._save_registry(new_registry)

        if not removed_from_disk and not removed_from_registry and model_info is None:
            return {
                "status": "not_found",
                "model_id": model_id,
            }

        return {
            "status": "uninstalled",
            "model_id": model_id,
            "removed_from_disk": removed_from_disk,
            "removed_from_registry": removed_from_registry,
        }

    def discover_models(
            self,
            search: str = "",
            limit: int = 20,
            task: str | None = None,
            only_diffusers: bool = False,
            only_sdxl: bool = False,
            installed_only: bool = False,
    ) -> list[dict]:
        limit = max(1, min(limit, 50))
        registry = self._load_registry_safe()
        installed_ids = {item.get("id", "") for item in registry}

        if installed_only:
            results: list[dict] = []

            for item in registry:
                model_id = item.get("id", "")
                if not model_id:
                    continue

                tags: list[str] = []
                pipeline_type = item.get("pipeline_type")

                if pipeline_type == "sdxl":
                    tags.append("sdxl")
                elif pipeline_type == "sd":
                    tags.append("sd")

                results.append({
                    "id": model_id,
                    "name": item.get("name", model_id.split("/")[-1]),
                    "author": model_id.split("/")[0] if "/" in model_id else None,
                    "downloads": None,
                    "likes": None,
                    "pipeline_tag": "text-to-image",
                    "tags": tags,
                    "installed": True,
                })

            return results[:limit]

        models = self.hf_api.list_models(
            search=search.strip() or None,
            sort="downloads",
            limit=limit * 3,
        )

        results: list[dict] = []

        for model in models:
            model_id = getattr(model, "id", "") or ""
            if not model_id:
                continue

            pipeline_tag = getattr(model, "pipeline_tag", None)
            tags = list(getattr(model, "tags", []) or [])
            installed = model_id in installed_ids

            if task and pipeline_tag != task:
                continue

            if only_diffusers and "diffusers" not in tags:
                continue

            if only_sdxl:
                lower_id = model_id.lower()
                lower_tags = [t.lower() for t in tags]
                is_sdxl = (
                        "sdxl" in lower_id or
                        "xl" in lower_id or
                        any("sdxl" in t for t in lower_tags)
                )
                if not is_sdxl:
                    continue

            results.append({
                "id": model_id,
                "name": model_id.split("/")[-1],
                "author": getattr(model, "author", None),
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
                "pipeline_tag": pipeline_tag,
                "tags": tags[:20],
                "installed": installed,
            })

            if len(results) >= limit:
                break

        return results

    def unload_model(self, model_id: str | None = None, unload_all: bool = False) -> dict:
        print(f"[MODEL] Unload requested | model_id={model_id} | unload_all={unload_all}")

        unloaded_models: list[str] = []

        if unload_all:
            unloaded_models = list(self._pipelines.keys())
            self._pipelines.clear()
            self.pipe = None

            gc.collect()
            self._empty_cuda_cache()

            return {
                "status": "unloaded",
                "scope": "all",
                "unloaded_models": unloaded_models,
            }

        if not model_id or not model_id.strip():
            raise ValueError("model_id is required when unload_all is false")

        model_id = model_id.strip()

        if model_id in self._pipelines:
            try:
                pipe_to_remove = self._pipelines[model_id]

                if self.pipe is pipe_to_remove:
                    self.pipe = None

                del self._pipelines[model_id]
                unloaded_models.append(model_id)
            except Exception:
                pass

        gc.collect()
        self._empty_cuda_cache()

        if not unloaded_models:
            return {
                "status": "not_loaded",
                "scope": "single",
                "model_id": model_id,
                "unloaded_models": [],
            }

        return {
            "status": "unloaded",
            "scope": "single",
            "model_id": model_id,
            "unloaded_models": unloaded_models,
        }