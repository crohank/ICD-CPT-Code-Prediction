"""
FastAPI application for ICD-10 code prediction.

Endpoints:
    POST /predict         — predict ICD-10 codes from discharge text
    GET  /health          — health check
    GET  /model/info      — model metadata and label vocabulary

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictionRequest, PredictionResponse, CodePrediction,
    EvidenceToken, HealthResponse, ModelInfoResponse,
)
from .model_service import ModelService

# One shared loader for the whole process — warmed up in the lifespan hook below.
service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    service.load()
    yield
    # Cleanup (if needed)


app = FastAPI(
    title="ICD-10 Code Prediction API",
    description="Predict ICD-10 diagnosis codes from clinical discharge summaries. "
                "Uses an ensemble of TF-IDF (Model A) and Chunk-Based BERT with "
                "Label Attention (Model C).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Streamlit / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict ICD-10 codes from a discharge summary.

    Optionally set `explain=true` to include attention-based evidence
    showing which words triggered each predicted code.
    """
    start = time.perf_counter()

    try:
        raw_results = service.predict(
            text=request.text,
            top_n=request.top_n,
            threshold=request.threshold,
            explain=request.explain,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    threshold_used = request.threshold or service.threshold_ens
    elapsed_ms = (time.perf_counter() - start) * 1000

    predictions = []
    for r in raw_results:
        evidence = None
        if r.get('evidence'):
            evidence = [EvidenceToken(token=e['token'], weight=e['weight'])
                        for e in r['evidence']]
        predictions.append(CodePrediction(
            icd_code=r['icd_code'],
            probability=r['probability'],
            predicted=r['predicted'],
            evidence=evidence,
        ))

    return PredictionResponse(
        predictions=predictions,
        threshold_used=threshold_used,
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns loaded models and device info."""
    return HealthResponse(
        status="ok",
        models_loaded=service.models_loaded,
        device=service.device,
        num_labels=service.num_labels,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and ICD-10 label vocabulary."""
    return ModelInfoResponse(
        models=service.models_loaded,
        num_labels=service.num_labels,
        label_vocab=service.vocab,
        thresholds={
            'model_a': service.threshold_a,
            'model_c': service.threshold_c,
            'ensemble': service.threshold_ens,
        },
        ensemble_weight=service.ensemble_weight,
    )
