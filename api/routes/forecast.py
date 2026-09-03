from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.config import settings
from src.data.loader import load_calendar, load_prices
from src.features.pipeline import FEATURE_COLUMNS
from src.forecasting.lightgbm_model import load_native_model
from src.forecasting.predictor import build_future_frame, recursive_predict
from src.forecasting.uncertainty import add_prediction_intervals

router = APIRouter()


def _load_assets():
    prepared_path = settings.processed_dir / "m5_prepared.csv"
    model_path = settings.models_dir / "lightgbm_model.txt"
    residuals_path = settings.metrics_dir / "validation_residuals.csv"
    if not prepared_path.exists():
        raise HTTPException(status_code=404, detail="Prepared data is missing.")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Trained model is missing.")
    history = pd.read_csv(prepared_path, parse_dates=["date"])
    model = load_native_model(model_path)
    residuals = pd.read_csv(residuals_path)["residual"] if residuals_path.exists() else []
    return history, model, residuals


def _forecast(series_id: str, horizon: int) -> list[dict]:
    if horizon < 1 or horizon > 365:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 365")
    history, model, residuals = _load_assets()
    if series_id not in set(history["series_id"].astype(str)):
        raise HTTPException(status_code=404, detail="Unknown series_id")
    calendar = load_calendar()
    keys = history[history["series_id"].astype(str) == series_id][["store_id", "item_id"]].drop_duplicates()
    prices = load_prices(series_keys=keys, data_mode="sample")
    future = build_future_frame(history, calendar, prices, horizon, [series_id])
    prediction = recursive_predict(model, history, future, FEATURE_COLUMNS)
    prediction = add_prediction_intervals(prediction, residuals, settings.prediction_interval)
    prediction["date"] = pd.to_datetime(prediction["date"]).dt.strftime("%Y-%m-%d")
    return prediction[["date", "series_id", "forecast", "lower_bound", "upper_bound"]].to_dict(orient="records")


@router.get("/forecast/{series_id}")
def forecast_by_id(series_id: str, horizon: int = 28) -> dict:
    return {"series_id": series_id, "horizon": horizon, "forecast": _forecast(series_id, horizon)}


@router.post("/forecast")
async def forecast_post(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    series_id = str(payload.get("series_id", "")).strip()
    horizon = payload.get("horizon", settings.forecast_horizon)
    if not series_id:
        raise HTTPException(status_code=400, detail="series_id is required")
    if not isinstance(horizon, int):
        raise HTTPException(status_code=400, detail="horizon must be an integer")
    return {"series_id": series_id, "horizon": horizon, "forecast": _forecast(series_id, horizon)}
