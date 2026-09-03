from _bootstrap import ROOT

import json
import pandas as pd

from src.config import settings
from src.features.pipeline import FEATURE_COLUMNS
from src.forecasting.lightgbm_model import feature_importance_frame, save_native_model
from src.forecasting.trainer import train_with_holdout
from src.forecasting.uncertainty import add_prediction_intervals, average_interval_width, interval_coverage
from src.utils.helpers import write_json
from src.utils.paths import ensure_directories


def main():
    path = settings.processed_dir / "m5_prepared.csv"
    if not path.exists():
        raise FileNotFoundError("Prepared data is missing. Run scripts/prepare_data.py first.")
    frame = pd.read_csv(path, parse_dates=["date"])
    result = train_with_holdout(frame, horizon=settings.forecast_horizon)
    scored = result["validation"].copy()
    residuals = scored["residual"].copy()
    validation_dates = sorted(pd.to_datetime(scored["date"].unique()))
    split_pos = max(1, len(validation_dates) // 2)
    calibration_dates = set(validation_dates[:split_pos])
    evaluation_dates = set(validation_dates[split_pos:])
    calibration_residuals = scored.loc[pd.to_datetime(scored["date"]).isin(calibration_dates), "residual"]
    interval_eval = scored.loc[pd.to_datetime(scored["date"]).isin(evaluation_dates)].copy()
    if interval_eval.empty:
        interval_eval = scored.copy()
    interval_frame = add_prediction_intervals(interval_eval, calibration_residuals, settings.prediction_interval)
    metrics = dict(result["metrics"])
    metrics["interval_coverage"] = interval_coverage(interval_frame["demand"], interval_frame["lower_bound"], interval_frame["upper_bound"])
    metrics["average_interval_width"] = average_interval_width(interval_frame["lower_bound"], interval_frame["upper_bound"])

    ensure_directories(settings.models_dir, settings.metrics_dir, settings.figures_dir)
    save_native_model(result["model"], settings.models_dir / "lightgbm_model.txt")
    (settings.models_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    metadata = {
        "model": "LightGBM",
        "forecast_horizon": settings.forecast_horizon,
        "prediction_interval": settings.prediction_interval,
        "training_rows": int(len(result["train"])),
        "validation_rows": int(len(scored)),
        "validation_start": str(pd.to_datetime(scored["date"]).min().date()),
        "validation_end": str(pd.to_datetime(scored["date"]).max().date()),
    }
    write_json(settings.models_dir / "model_metadata.json", metadata)
    write_json(settings.metrics_dir / "validation_metrics.json", metrics)
    scored[["series_id", "date", "residual"]].to_csv(settings.metrics_dir / "validation_residuals.csv", index=False)
    feature_importance_frame(result["model"]).to_csv(settings.metrics_dir / "feature_importance.csv", index=False)
    print("Validation metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved native model to {settings.models_dir / 'lightgbm_model.txt'}")


if __name__ == "__main__":
    main()
