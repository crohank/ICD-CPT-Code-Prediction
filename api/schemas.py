"""
Request and response shapes for the FastAPI layer.

Keeping these as Pydantic models gives you validation on `/predict` (min text
length, sensible `top_n`) and a stable JSON contract for any frontend or tests.
"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Input: a discharge summary text."""
    text: str = Field(..., description="Discharge summary text", min_length=10)
    top_n: int = Field(10, description="Number of top codes to return", ge=1, le=50)
    threshold: Optional[float] = Field(
        None, description="Custom decision threshold (uses tuned default if None)",
        ge=0.0, le=1.0,
    )
    explain: bool = Field(False, description="Include attention-based evidence per code")


class EvidenceToken(BaseModel):
    """A single token with its attention weight."""
    token: str
    weight: float


class CodePrediction(BaseModel):
    """A single predicted ICD-10 code with metadata."""
    icd_code: str
    description: str = ""
    probability: float
    predicted: bool
    evidence: Optional[list[EvidenceToken]] = None


class PredictionResponse(BaseModel):
    """API response with predicted codes."""
    predictions: list[CodePrediction]
    model_version: str = "ensemble_v1"
    threshold_used: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    models_loaded: list[str] = []
    device: str = "cpu"
    num_labels: int = 0


class ModelInfoResponse(BaseModel):
    """Model metadata."""
    models: list[str]
    num_labels: int
    label_vocab: list[str]
    thresholds: dict[str, float]
    ensemble_weight: Optional[float] = None
