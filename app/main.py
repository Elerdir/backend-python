from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_generation import router as generation_router
from app.api.routes_images import router as images_router
from app.api.routes_runtime import router as runtime_router  # 👈 NOVÉ
from app.api.routes_translation import router as translation_router

app = FastAPI(
    title="Image Studio Backend",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(generation_router)
app.include_router(images_router)
app.include_router(runtime_router)  # 👈 NOVÉ
app.include_router(translation_router)