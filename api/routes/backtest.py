from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.config import settings
from src.forecasting.backtesting import run_backtest, summarize_backtest

router = APIRouter()


@router.post("/backtest")
async def backtest(request: Request) -> dict:
    path = settings.processed_dir / "m5_prepared.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prepared data is missing.")
    payload = await request.json()
    horizon = int(payload.get("horizon", settings.forecast_horizon))
    folds = int(payload.get("folds", settings.n_backtest_folds))
    frame = pd.read_csv(path, parse_dates=["date"])
    try:
        results, _ = run_backtest(frame, horizon=horizon, n_folds=folds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": results.to_dict(orient="records"), "summary": summarize_backtest(results)}
