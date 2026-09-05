from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "demo" / "products.csv"


@router.get("/products")
def products(
    state_id: str | None = None,
    store_id: str | None = None,
    cat_id: str | None = None,
    dept_id: str | None = None,
) -> dict:
    if not PRODUCTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Demo product metadata is missing.",
        )

    frame = pd.read_csv(PRODUCTS_PATH)

    filters = {
        "state_id": state_id,
        "store_id": store_id,
        "cat_id": cat_id,
        "dept_id": dept_id,
    }

    for column, value in filters.items():
        if value and column in frame.columns:
            frame = frame[
                frame[column].astype(str) == str(value)
            ]

    records = frame.to_dict(orient="records")

    return {
        "count": len(records),
        "products": records,
    }