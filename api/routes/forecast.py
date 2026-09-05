from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "demo" / "forecasts"


def _forecast(series_id: str, horizon: int) -> list[dict]:
    if horizon < 1 or horizon > 28:
        raise HTTPException(
            status_code=400,
            detail="Demo horizon must be between 1 and 28 days.",
        )

    if not re.fullmatch(r"[A-Za-z0-9_]+", series_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid series_id.",
        )

    path = FORECAST_DIR / f"forecast_{series_id}.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Forecast is not available for this demo product.",
        )

    frame = pd.read_csv(path)

    required = [
        "date",
        "series_id",
        "forecast",
        "lower_bound",
        "upper_bound",
    ]

    missing = [c for c in required if c not in frame.columns]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast artifact is missing columns: {missing}",
        )

    frame = frame.head(horizon).copy()

    frame["date"] = (
        pd.to_datetime(frame["date"])
        .dt.strftime("%Y-%m-%d")
    )

    return frame[required].to_dict(orient="records")


@router.get("/forecast/{series_id}")
def forecast_by_id(
    series_id: str,
    horizon: int = 28,
) -> dict:
    forecast = _forecast(series_id, horizon)

    return {
        "series_id": series_id,
        "horizon": horizon,
        "forecast": forecast,
    }


@router.post("/forecast")
async def forecast_post(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Request body must be valid JSON.",
        ) from exc

    series_id = str(payload.get("series_id", "")).strip()
    horizon = payload.get("horizon", 28)

    if not series_id:
        raise HTTPException(
            status_code=400,
            detail="series_id is required.",
        )

    if not isinstance(horizon, int):
        raise HTTPException(
            status_code=400,
            detail="horizon must be an integer.",
        )

    forecast = _forecast(series_id, horizon)

    return {
        "series_id": series_id,
        "horizon": horizon,
        "forecast": forecast,
    }