from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.loader import check_expected_files


def validate_m5_data(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    raw_dir: Path | None = None,
) -> dict:
    raw_dir = Path(raw_dir or settings.raw_m5_dir)
    files = check_expected_files(raw_dir)
    missing_required = [name for name, exists in files.items() if not exists]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError("Sales data has no d_* demand columns.")
    if sales["id"].duplicated().any():
        raise ValueError("Duplicate sales series IDs detected.")
    if calendar["d"].duplicated().any():
        raise ValueError("Duplicate calendar day keys detected.")
    if not prices.empty and prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Duplicate sell-price keys detected.")
    if (sales[day_cols] < 0).any().any():
        raise ValueError("Negative demand values detected.")
    if not calendar["date"].is_monotonic_increasing:
        calendar = calendar.sort_values("date")
    expected_dates = pd.date_range(calendar["date"].min(), calendar["date"].max(), freq="D")
    missing_dates = len(expected_dates.difference(pd.DatetimeIndex(calendar["date"])))

    return {
        "expected_files": files,
        "missing_expected_files": missing_required,
        "sales_shape": list(sales.shape),
        "calendar_shape": list(calendar.shape),
        "prices_shape": list(prices.shape),
        "sales_memory_mb": round(float(sales.memory_usage(deep=True).sum() / 1024**2), 2),
        "calendar_memory_mb": round(float(calendar.memory_usage(deep=True).sum() / 1024**2), 2),
        "prices_memory_mb": round(float(prices.memory_usage(deep=True).sum() / 1024**2), 2),
        "sales_missing_values": int(sales.isna().sum().sum()),
        "calendar_missing_values": int(calendar.isna().sum().sum()),
        "prices_missing_values": int(prices.isna().sum().sum()),
        "calendar_missing_dates": int(missing_dates),
        "date_range": [str(calendar["date"].min().date()), str(calendar["date"].max().date())],
        "number_of_series": int(sales["id"].nunique()),
        "number_of_stores": int(sales["store_id"].nunique()),
        "number_of_products": int(sales["item_id"].nunique()),
    }
