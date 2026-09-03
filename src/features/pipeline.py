from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from src.features.calendar_features import add_calendar_features
from src.features.lag_features import add_lag_features
from src.features.price_features import add_price_features
from src.features.rolling_features import add_rolling_features

IDENTIFIER_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "series_id"]

FEATURE_COLUMNS = [
    "day_of_week", "week", "month", "quarter", "year", "is_weekend",
    "is_month_start", "is_month_end", "event_indicator", "snap_CA", "snap_TX", "snap_WI",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28", "lag_56",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_mean_56",
    "rolling_std_7", "rolling_std_28", "current_price", "previous_price",
    "price_change", "price_change_pct", "rolling_average_price", "relative_price",
    "item_id_code", "dept_id_code", "cat_id_code", "store_id_code", "state_id_code", "series_id_code",
]


def _stable_code(value: object) -> int:
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


def add_identifier_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in IDENTIFIER_COLUMNS:
        if col in out.columns:
            out[f"{col}_code"] = out[col].astype(str).map(_stable_code).astype("uint32")
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(frame)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_price_features(out)
    out = add_identifier_features(out)
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    featured = build_features(frame)
    usable = featured.dropna(subset=["demand", "lag_56"]).copy()
    x = usable[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = usable["demand"].astype(float)
    return x, y, usable
