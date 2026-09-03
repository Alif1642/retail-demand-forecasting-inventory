from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.pipeline import FEATURE_COLUMNS, build_features


def recursive_predict(model, history: pd.DataFrame, future: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    history_part = history.copy()
    future_part = future.copy()
    future_part["demand"] = np.nan
    combined = pd.concat([history_part, future_part], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"])
    future_dates = sorted(pd.to_datetime(future_part["date"].unique()))

    predictions = []
    for current_date in future_dates:
        featured = build_features(combined)
        mask = (featured["date"] == current_date) & featured["demand"].isna()
        if not mask.any():
            continue
        x = featured.loc[mask, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        values = np.maximum(0.0, np.asarray(model.predict(x), dtype=float))
        target_indices = featured.index[mask]
        combined.loc[target_indices, "demand"] = values
        current_rows = featured.loc[mask, ["series_id", "date"]].copy()
        current_rows["forecast"] = values
        predictions.append(current_rows)

    if not predictions:
        return pd.DataFrame(columns=["series_id", "date", "forecast"])
    return pd.concat(predictions, ignore_index=True).sort_values(["series_id", "date"])


def build_future_frame(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    if series_ids is None:
        series_ids = history["series_id"].drop_duplicates().astype(str).tolist()
    selected = history[history["series_id"].astype(str).isin(series_ids)]
    if selected.empty:
        raise ValueError("No matching series found in prepared data.")

    last_date = selected["date"].max()
    future_calendar = calendar[pd.to_datetime(calendar["date"]) > last_date].sort_values("date").head(horizon).copy()
    if len(future_calendar) < horizon:
        raise ValueError("Calendar does not contain enough future dates for the requested horizon.")

    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    cal_cols = [c for c in cal_cols if c in future_calendar.columns]
    future_calendar = future_calendar[cal_cols]

    meta_cols = ["series_id", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    metadata = selected.sort_values("date").groupby("series_id", observed=True).tail(1)[meta_cols]
    metadata = metadata[metadata["series_id"].astype(str).isin(series_ids)]
    metadata["_key"] = 1
    future_calendar["_key"] = 1
    future = metadata.merge(future_calendar, on="_key").drop(columns="_key")
    if not prices.empty:
        future = future.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    else:
        future["sell_price"] = np.nan
    future["demand"] = np.nan
    return future.sort_values(["series_id", "date"]).reset_index(drop=True)
