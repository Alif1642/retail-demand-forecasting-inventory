from __future__ import annotations

import pandas as pd

from src.data.schema import SALES_ID_COLUMNS


def normalize_series_id(raw_id: str) -> str:
    value = str(raw_id)
    for suffix in ("_evaluation", "_validation"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def sales_wide_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    day_cols = [col for col in sales.columns if col.startswith("d_")]
    if not day_cols:
        raise ValueError("Sales data does not contain M5 d_* columns.")
    long = sales.melt(
        id_vars=SALES_ID_COLUMNS,
        value_vars=day_cols,
        var_name="d",
        value_name="demand",
    )
    long["series_id"] = long["id"].astype(str).map(normalize_series_id)
    long["demand"] = pd.to_numeric(long["demand"], downcast="integer")
    return long


def merge_calendar(long_sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    calendar_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    available = [c for c in calendar_cols if c in calendar.columns]
    merged = long_sales.merge(calendar[available], on="d", how="left", validate="many_to_one")
    return merged


def merge_prices(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        out = frame.copy()
        out["sell_price"] = pd.NA
        return out
    return frame.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )


def prepare_m5_long(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    frame = sales_wide_to_long(sales)
    frame = merge_calendar(frame, calendar)
    frame = merge_prices(frame, prices)
    frame = frame.sort_values(["series_id", "date"]).reset_index(drop=True)
    return frame
