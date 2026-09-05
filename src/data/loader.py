from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import settings
from src.data.schema import REQUIRED_M5_FILES, SALES_ID_COLUMNS


def _downcast_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.select_dtypes(include=["integer"]).columns:
        frame[col] = pd.to_numeric(frame[col], downcast="integer")
    for col in frame.select_dtypes(include=["float"]).columns:
        frame[col] = pd.to_numeric(frame[col], downcast="float")
    return frame


def _as_categories(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = frame[col].astype("category")
    return frame


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required M5 file is missing: {path}. Download the M5 dataset and place the CSV files in {settings.raw_m5_dir}."
        )
    return path


def load_calendar(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir or settings.raw_m5_dir)
    path = _require_file(raw_dir / "calendar.csv")
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = _as_categories(
        frame,
        ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"],
    )
    return _downcast_numeric(frame)


def _sales_path(raw_dir: Path) -> Path:
    evaluation = raw_dir / "sales_train_evaluation.csv"
    validation = raw_dir / "sales_train_validation.csv"
    if evaluation.exists():
        return evaluation
    if validation.exists():
        return validation
    raise FileNotFoundError(
        f"Neither sales training file exists in {raw_dir}. Expected an M5 sales CSV."
    )


def load_sales(
    raw_dir: Path | None = None,
    data_mode: str | None = None,
    max_series: int | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir or settings.raw_m5_dir)
    data_mode = (data_mode or settings.data_mode).lower()
    max_series = int(max_series or settings.max_series)
    path = _sales_path(raw_dir)
    nrows = max_series if data_mode == "sample" else None
    frame = pd.read_csv(path, nrows=nrows)
    frame = _as_categories(frame, [c for c in SALES_ID_COLUMNS if c != "id"])
    day_cols = [c for c in frame.columns if c.startswith("d_")]
    frame[day_cols] = frame[day_cols].apply(pd.to_numeric, downcast="integer")
    return frame


def load_prices(
    raw_dir: Path | None = None,
    series_keys: pd.DataFrame | None = None,
    data_mode: str | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir or settings.raw_m5_dir)
    data_mode = (data_mode or settings.data_mode).lower()

    full_path = raw_dir / "sell_prices.csv"
    sample_path = raw_dir / "sell_prices_sample.csv"

    # In sample/deployment mode, use the lightweight pre-filtered file.
    if data_mode == "sample" and sample_path.exists():
        frame = pd.read_csv(sample_path)

        if series_keys is not None:
            valid_pairs = set(
                series_keys["store_id"]
                .astype(str)
                .str.cat(series_keys["item_id"].astype(str), sep="::")
            )

            pair_key = (
                frame["store_id"]
                .astype(str)
                .str.cat(frame["item_id"].astype(str), sep="::")
            )

            frame = frame.loc[pair_key.isin(valid_pairs)].copy()

    else:
        path = _require_file(full_path)

        if data_mode == "sample" and series_keys is not None:
            valid_pairs = set(
                series_keys["store_id"]
                .astype(str)
                .str.cat(series_keys["item_id"].astype(str), sep="::")
            )

            chunks = []

            for chunk in pd.read_csv(path, chunksize=500_000):
                pair_key = (
                    chunk["store_id"]
                    .astype(str)
                    .str.cat(chunk["item_id"].astype(str), sep="::")
                )

                filtered = chunk.loc[pair_key.isin(valid_pairs)]

                if not filtered.empty:
                    chunks.append(filtered)

            frame = (
                pd.concat(chunks, ignore_index=True)
                if chunks
                else pd.DataFrame(
                    columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]
                )
            )
        else:
            frame = pd.read_csv(path)

    frame = _as_categories(frame, ["store_id", "item_id"])
    return _downcast_numeric(frame)


def load_m5_data(
    raw_dir: Path | None = None,
    data_mode: str | None = None,
    max_series: int | None = None,
) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir or settings.raw_m5_dir)
    sales = load_sales(raw_dir, data_mode=data_mode, max_series=max_series)
    calendar = load_calendar(raw_dir)
    keys = sales[["store_id", "item_id"]].drop_duplicates()
    prices = load_prices(raw_dir, series_keys=keys, data_mode=data_mode)
    return {"sales": sales, "calendar": calendar, "prices": prices}


def check_expected_files(raw_dir: Path | None = None) -> dict[str, bool]:
    raw_dir = Path(raw_dir or settings.raw_m5_dir)
    return {name: (raw_dir / name).exists() for name in REQUIRED_M5_FILES}

