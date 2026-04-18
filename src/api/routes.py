from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import traceback

from src.schemas.request import FinalRegenRequest, ReasoningRequest, StepRegenRequest
from src.schemas.response import ReasoningResponse, RegenerationType
from src.reasoning.pipeline import run_reasoning_pipeline, run_step_regen_pipeline, run_final_regen_pipeline

router = APIRouter()


@router.post("/reason", response_model=ReasoningResponse)
def reason(request: ReasoningRequest):
    try:
        return run_reasoning_pipeline(request)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reason/step", response_model=ReasoningResponse)
def regen_step(request: StepRegenRequest):
    try:
        response = run_step_regen_pipeline(request)
        response.step_regenerated[request.step_to_regenerate_index] = RegenerationType.MANUAL
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reason/final", response_model=ReasoningResponse)
def regen_final(request: FinalRegenRequest):
    try:
        response = run_final_regen_pipeline(request)
        response.final_regenerated = RegenerationType.MANUAL
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/")
def serve_frontend():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "index.html"
    return FileResponse(file_path)