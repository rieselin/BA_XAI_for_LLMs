from pydantic import BaseModel, Field
from typing import List

from src.schemas.response import RegenerationType, TokenConfidence


class ReasoningRequest(BaseModel):
    input: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1, le=10)

class RegenerationRequest(ReasoningRequest):
    steps: List[str]
    tokens: List[TokenConfidence]
    step_regenerated: List[RegenerationType]

class StepRegenRequest(RegenerationRequest):
    step_to_regenerate_index: int
    final_answer: str
    final_regenerated: RegenerationType

class FinalRegenRequest(RegenerationRequest):
    pass