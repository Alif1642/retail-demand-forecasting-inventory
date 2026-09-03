from __future__ import annotations

import pandas as pd


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    iso = out["date"].dt.isocalendar()
    out["day_of_week"] = out["date"].dt.dayofweek.astype("int8")
    out["week"] = iso.week.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out["quarter"] = out["date"].dt.quarter.astype("int8")
    out["year"] = out["date"].dt.year.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = out["date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")
    out["event_indicator"] = 0
    for col in ("event_name_1", "event_name_2"):
        if col in out.columns:
            out["event_indicator"] = (out["event_indicator"] | out[col].notna()).astype("int8")
    for col in ("snap_CA", "snap_TX", "snap_WI"):
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype("int8")
    return out
