from pydantic import BaseModel, Field
from typing import List, Optional


class ReasoningRequest(BaseModel):
    input: str
    include_cot: bool = True
    include_confidence: bool = True
    threshold: Optional[float] = None      # 0.0–1.0, None = no auto-regen
    max_attempts: int = Field(default=3, ge=1, le=10)


class StepRegenRequest(BaseModel):
    input: str
    prior_steps: List[str]
    step_number: int                       # 1-indexed
    max_attempts: int = Field(default=3, ge=1, le=10)
    threshold: Optional[float] = None     # if provided, loop until above it or exhausted