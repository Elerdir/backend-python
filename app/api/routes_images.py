import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

OUTPUT_DIR = Path(os.getenv("APP_DATA_DIR", "data")) / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/images/{filename}")
async def get_generated_image(filename: str):
    # Prevent path traversal: strip to basename only and reject anything
    # that still contains a separator (e.g. "../secret", "sub/file.png").
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = OUTPUT_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)