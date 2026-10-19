"""FastAPI app (Phase 14).

Start (from the project root, virtual environment active):
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Interactive test page:  http://127.0.0.1:8000/docs

Endpoints
    GET  /health    is the model loaded?
    POST /predict   {"text": "..."}  ->  {"prediction", "confidence", "language", "important_words"}
                    add "details": true for highlights, method info and timings
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.model_service import ModelService
from backend.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hate_api")

service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model from %s ...", service.model_dir)
        service.load()
        logger.info("Model ready on %s (%.1f s)", service.device, service.load_seconds)
    except Exception as exc:  # noqa: BLE001  - keep the server up so /health can explain the problem
        logger.error("Could not load the model: %s", exc)
    yield


app = FastAPI(title="Multilingual hate / abusive speech detection API", version="1.0.0", lifespan=lifespan)

_origins = os.getenv("HATE_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _origins], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root() -> dict:
    return {"service": "hate-speech-detection", "docs": "/docs", "health": "/health", "predict": "POST /predict"}


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return service.info()


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(req: PredictRequest) -> dict:
    # plain `def` (not async): FastAPI runs it in a worker thread, so the slow model call does not block the server
    if not service.ready:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {service.error or 'still loading'}")
    try:
        return service.analyze(req.text, explain=req.explain, details=req.details)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
