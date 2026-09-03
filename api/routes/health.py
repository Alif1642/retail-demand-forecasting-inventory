from __future__ import annotations

from fastapi import APIRouter

from src.config import settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "data_mode": settings.data_mode}


@router.get("/models")
def models() -> dict:
    model_path = settings.models_dir / "lightgbm_model.txt"
    return {"lightgbm": {"available": model_path.exists(), "path": str(model_path)}}


@router.get("/metrics")
def metrics() -> dict:
    path = settings.metrics_dir / "validation_metrics.json"
    if not path.exists():
        return {"available": False, "message": "Run model training to generate metrics."}
    import json
    return {"available": True, "metrics": json.loads(path.read_text(encoding="utf-8"))}
