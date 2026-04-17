from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from src.schemas.request import ReasoningRequest, StepRegenRequest
from src.schemas.response import ReasoningResponse
from src.reasoning.pipeline import run_reasoning_pipeline, run_step_regen_pipeline

router = APIRouter()


@router.post("/reason", response_model=ReasoningResponse)
def reason(request: ReasoningRequest):
    try:
        return run_reasoning_pipeline(request)
    except Exception as e:
        import traceback
        traceback.print_exc()          # ← prints full stack to your server console
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reason/step", response_model=ReasoningResponse)
def regen_step(request: StepRegenRequest):
    try:
        return run_step_regen_pipeline(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reason/final", response_model=ReasoningResponse)
def regen_final(request: dict):
    try:
        from src.reasoning.pipeline import run_final_regen_pipeline
        return run_final_regen_pipeline(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def serve_frontend():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "index.html"
    return FileResponse(file_path)