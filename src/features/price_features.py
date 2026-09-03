from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["series_id", "date"]).copy()
    out["current_price"] = pd.to_numeric(out.get("sell_price"), errors="coerce")
    grouped = out.groupby("series_id", observed=True)["current_price"]
    out["previous_price"] = grouped.shift(1)
    out["price_change"] = out["current_price"] - out["previous_price"]
    denom = out["previous_price"].replace(0, np.nan)
    out["price_change_pct"] = out["price_change"] / denom
    out["rolling_average_price"] = grouped.transform(
        lambda series: series.rolling(28, min_periods=1).mean()
    )
    avg_denom = out["rolling_average_price"].replace(0, np.nan)
    out["relative_price"] = out["current_price"] / avg_denom
    return out
