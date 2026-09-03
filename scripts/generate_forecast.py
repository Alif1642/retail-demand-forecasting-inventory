from _bootstrap import ROOT

import argparse
import pandas as pd

from src.config import settings
from src.data.loader import load_calendar, load_prices
from src.features.pipeline import FEATURE_COLUMNS
from src.forecasting.lightgbm_model import load_native_model
from src.forecasting.predictor import build_future_frame, recursive_predict
from src.forecasting.uncertainty import add_prediction_intervals
from src.utils.paths import ensure_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--horizon", type=int, default=settings.forecast_horizon)
    args = parser.parse_args()

    prepared_path = settings.processed_dir / "m5_prepared.csv"
    model_path = settings.models_dir / "lightgbm_model.txt"
    if not prepared_path.exists():
        raise FileNotFoundError("Prepared data is missing. Run data preparation first.")
    history = pd.read_csv(prepared_path, parse_dates=["date"])
    if args.series_id not in set(history["series_id"].astype(str)):
        raise ValueError(f"Unknown series_id: {args.series_id}")
    model = load_native_model(model_path)
    calendar = load_calendar()
    keys = history[history["series_id"].astype(str) == args.series_id][["store_id", "item_id"]].drop_duplicates()
    prices = load_prices(series_keys=keys, data_mode="sample")
    future = build_future_frame(history, calendar, prices, args.horizon, [args.series_id])
    prediction = recursive_predict(model, history, future, FEATURE_COLUMNS)
    residual_path = settings.metrics_dir / "validation_residuals.csv"
    residuals = pd.read_csv(residual_path)["residual"] if residual_path.exists() else []
    prediction = add_prediction_intervals(prediction, residuals, settings.prediction_interval)
    prediction = prediction[["date", "series_id", "forecast", "lower_bound", "upper_bound"]]
    ensure_directories(settings.predictions_dir)
    output = settings.predictions_dir / f"forecast_{args.series_id}.csv"
    prediction.to_csv(output, index=False)
    print(prediction.to_string(index=False))
    print(f"Saved forecast to {output}")


if __name__ == "__main__":
    main()
