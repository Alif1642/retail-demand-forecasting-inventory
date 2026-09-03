from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    raw_m5_dir: Path = PROJECT_ROOT / "data" / "raw" / "m5"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    predictions_dir: Path = PROJECT_ROOT / "data" / "predictions"
    models_dir: Path = PROJECT_ROOT / "models"
    metrics_dir: Path = PROJECT_ROOT / "results" / "metrics"
    figures_dir: Path = PROJECT_ROOT / "results" / "figures"
    data_mode: str = os.getenv("DATA_MODE", "sample").lower()
    max_series: int = _env_int("MAX_SERIES", 100)
    forecast_horizon: int = _env_int("FORECAST_HORIZON", 28)
    n_backtest_folds: int = _env_int("N_BACKTEST_FOLDS", 3)
    prediction_interval: float = _env_float("PREDICTION_INTERVAL", 0.90)
    default_service_level: float = _env_float("DEFAULT_SERVICE_LEVEL", 0.95)
    default_lead_time_days: int = _env_int("DEFAULT_LEAD_TIME_DAYS", 7)

    def validate(self) -> None:
        if self.data_mode not in {"sample", "full"}:
            raise ValueError("DATA_MODE must be 'sample' or 'full'.")
        if self.max_series < 1:
            raise ValueError("MAX_SERIES must be positive.")
        if self.forecast_horizon < 1:
            raise ValueError("FORECAST_HORIZON must be positive.")
        if not 0 < self.prediction_interval < 1:
            raise ValueError("PREDICTION_INTERVAL must be between 0 and 1.")


settings = Settings()
