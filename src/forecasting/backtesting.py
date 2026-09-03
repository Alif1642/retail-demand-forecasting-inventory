from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.evaluator import evaluate_forecasts
from src.features.pipeline import FEATURE_COLUMNS
from src.forecasting.predictor import recursive_predict
from src.forecasting.seasonal_naive import SeasonalNaive
from src.forecasting.trainer import fit_model
from src.forecasting.uncertainty import add_prediction_intervals, interval_coverage


def rolling_origins(frame: pd.DataFrame, horizon: int, n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted(pd.to_datetime(frame["date"].unique()))
    required = horizon * n_folds + 57
    if len(dates) < required:
        raise ValueError(f"Need at least {required} dates for {n_folds} folds and lag warm-up.")
    folds = []
    for fold_idx in range(n_folds):
        end_pos = len(dates) - horizon * (n_folds - fold_idx - 1)
        start_pos = end_pos - horizon
        folds.append((dates[start_pos], dates[end_pos - 1]))
    return folds


def _seasonal_fold(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    model = SeasonalNaive(7)
    rows = []
    for series_id, group in valid.groupby("series_id", observed=True, sort=False):
        hist = train.loc[train["series_id"] == series_id].sort_values("date")
        target = group.sort_values("date")
        forecast = model.forecast_values(hist["demand"], len(target))
        out = target[["series_id", "date", "demand"]].copy()
        out["forecast"] = forecast
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def run_backtest(frame: pd.DataFrame, horizon: int = 28, n_folds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    residual_pool = []
    detailed = []
    for fold_number, (start_date, end_date) in enumerate(rolling_origins(frame, horizon, n_folds), start=1):
        train = frame[pd.to_datetime(frame["date"]) < start_date].copy()
        valid = frame[(pd.to_datetime(frame["date"]) >= start_date) & (pd.to_datetime(frame["date"]) <= end_date)].copy()

        baseline_scored = _seasonal_fold(train, valid)
        baseline_metrics = evaluate_forecasts(baseline_scored, history=train)
        for key, value in baseline_metrics.items():
            pass
        records.append({"model": "seasonal_naive", "fold": fold_number, **baseline_metrics, "interval_coverage": float("nan")})
        baseline_scored["model"] = "seasonal_naive"
        baseline_scored["fold"] = fold_number
        detailed.append(baseline_scored)

        model, _ = fit_model(train)
        future = valid.copy()
        future["demand"] = np.nan
        preds = recursive_predict(model, train, future, FEATURE_COLUMNS)
        scored = valid[["series_id", "date", "demand"]].merge(preds, on=["series_id", "date"], how="inner")
        metrics = evaluate_forecasts(scored, history=train)
        if residual_pool:
            interval_frame = add_prediction_intervals(scored, residual_pool)
            coverage = interval_coverage(interval_frame["demand"], interval_frame["lower_bound"], interval_frame["upper_bound"])
        else:
            # The first fold has no earlier validation fold. Use the first half of its
            # validation dates for interval calibration and evaluate coverage on the
            # second half, keeping calibration and coverage rows separate.
            fold_dates = sorted(pd.to_datetime(scored["date"].unique()))
            split_pos = max(1, len(fold_dates) // 2)
            calibration_dates = set(fold_dates[:split_pos])
            evaluation_dates = set(fold_dates[split_pos:])
            calibration_residuals = (
                scored.loc[pd.to_datetime(scored["date"]).isin(calibration_dates), "demand"]
                - scored.loc[pd.to_datetime(scored["date"]).isin(calibration_dates), "forecast"]
            )
            interval_target = scored.loc[pd.to_datetime(scored["date"]).isin(evaluation_dates)].copy()
            if interval_target.empty:
                interval_target = scored.copy()
            interval_frame = add_prediction_intervals(interval_target, calibration_residuals)
            coverage = interval_coverage(interval_frame["demand"], interval_frame["lower_bound"], interval_frame["upper_bound"])
        records.append({"model": "lightgbm", "fold": fold_number, **metrics, "interval_coverage": coverage})
        scored["model"] = "lightgbm"
        scored["fold"] = fold_number
        detailed.append(scored)
        residual_pool.extend((scored["demand"] - scored["forecast"]).tolist())

    return pd.DataFrame(records), pd.concat(detailed, ignore_index=True)


def summarize_backtest(results: pd.DataFrame) -> dict:
    summary = {}
    for model, group in results.groupby("model"):
        metric_cols = [c for c in ["MAE", "WMAPE", "RMSSE", "Bias", "interval_coverage"] if c in group]
        summary[model] = {
            "mean": group[metric_cols].mean(numeric_only=True).to_dict(),
            "std": group[metric_cols].std(numeric_only=True).to_dict(),
            "best_fold_by_MAE": int(group.loc[group["MAE"].idxmin(), "fold"]),
            "worst_fold_by_MAE": int(group.loc[group["MAE"].idxmax(), "fold"]),
        }
    return summary
