from pydantic import BaseModel
from typing import List
from enum import Enum

class TokenConfidence(BaseModel):
    token: str
    prob: float

class RegenerationType(Enum):
    NOT_REGENERATED = 0
    MANUAL = 1
    AUTO = 2

class StepConfidence(BaseModel):
    step: str
    tokens: List[TokenConfidence]
    mean_confidence: float

class ReasoningResponse(BaseModel):
    final_answer: str
    steps: List[str]

    step_confidences: List[StepConfidence]
    final_answer_confidence: StepConfidence

    step_regenerated: List[RegenerationType]
    final_regenerated: RegenerationType