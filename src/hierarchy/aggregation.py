from __future__ import annotations

import pandas as pd


def aggregate_forecasts(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    level_map = {
        "total": [],
        "state": ["state_id"],
        "store": ["store_id"],
        "category": ["cat_id"],
        "department": ["dept_id"],
        "sku_store": ["item_id", "store_id"],
    }
    if level not in level_map:
        raise ValueError(f"Unsupported hierarchy level: {level}")
    group_cols = level_map[level] + ["date"]
    if level == "total":
        out = frame.groupby("date", as_index=False)["forecast"].sum()
        out["level"] = "total"
        return out
    return frame.groupby(group_cols, as_index=False, observed=True)["forecast"].sum()
