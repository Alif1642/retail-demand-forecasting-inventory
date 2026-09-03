from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.evaluator import evaluate_forecasts
from src.features.pipeline import FEATURE_COLUMNS, training_matrix
from src.forecasting.lightgbm_model import build_model
from src.forecasting.predictor import recursive_predict


def chronological_split(frame: pd.DataFrame, horizon: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(frame["date"].dropna().unique()))
    if len(dates) <= horizon + 56:
        raise ValueError("Not enough history for a chronological split after feature warm-up.")
    cutoff = dates[-horizon]
    train = frame[pd.to_datetime(frame["date"]) < cutoff].copy()
    valid = frame[pd.to_datetime(frame["date"]) >= cutoff].copy()
    return train, valid


def fit_model(frame: pd.DataFrame, params: dict | None = None):
    x, y, usable = training_matrix(frame)
    if x.empty:
        raise ValueError("No training rows remain after lag warm-up.")
    model = build_model(params)
    model.fit(x, y)
    return model, usable


def train_with_holdout(frame: pd.DataFrame, horizon: int = 28, params: dict | None = None) -> dict:
    train, valid_actual = chronological_split(frame, horizon)
    model, _ = fit_model(train, params)
    future = valid_actual.copy()
    future["demand"] = np.nan
    predicted = recursive_predict(model, train, future, FEATURE_COLUMNS)
    actual = valid_actual[["series_id", "date", "demand"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])
    scored = actual.merge(predicted, on=["series_id", "date"], how="inner")
    metrics = evaluate_forecasts(scored, history=train)
    scored["residual"] = scored["demand"] - scored["forecast"]
    return {"model": model, "train": train, "validation": scored, "metrics": metrics}
