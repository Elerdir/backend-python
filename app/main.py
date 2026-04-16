from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_health import router as health_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_generation import router as generation_router, job_service
from app.api.routes_images import router as images_router
from app.api.routes_runtime import router as runtime_router
from app.api.routes_translation import router as translation_router

# Maximum allowed upload body size (20 MB).  Multipart generation jobs send
# image files, but anything above this threshold is almost certainly abuse.
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful shutdown: signal the job-worker thread to stop so it can
    # finish cleanup rather than being killed mid-job by the OS.
    job_service.stop_worker()


app = FastAPI(
    title="Image Studio Backend",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject requests whose Content-Length exceeds the allowed maximum."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            size = int(content_length)
        except ValueError:
            size = 0
        if size > _MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"},
            )
    return await call_next(request)


app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(generation_router)
app.include_router(images_router)
app.include_router(runtime_router)
app.include_router(translation_router)