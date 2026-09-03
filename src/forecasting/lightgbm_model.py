from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd


DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 350,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def build_model(params: dict | None = None) -> lgb.LGBMRegressor:
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return lgb.LGBMRegressor(**merged)


def save_native_model(model: lgb.LGBMRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_native_model(path: Path) -> lgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return lgb.Booster(model_file=str(path))


def feature_importance_frame(model: lgb.LGBMRegressor) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
