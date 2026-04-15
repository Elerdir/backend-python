from fastapi import APIRouter
from app.api.routes_generation import shared_generation_service

router = APIRouter()


@router.get("/runtime-info")
async def get_runtime_info():
    return shared_generation_service.get_runtime_info()