from pydantic import BaseModel
from typing import List, Optional


class TokenConfidence(BaseModel):
    token: str
    prob: float


class ConfidenceSummary(BaseModel):
    mean: float
    min: float


class ReasoningResponse(BaseModel):
    final_answer: str
    steps: List[str]

    confidence: Optional[ConfidenceSummary]
    step_confidence: Optional[List[float]]

    tokens: Optional[List[TokenConfidence]]