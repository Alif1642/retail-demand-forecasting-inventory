from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.config import settings

router = APIRouter()


@router.get("/products")
def products(limit: int = Query(100, ge=1, le=1000)) -> dict:
    path = settings.processed_dir / "m5_prepared.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prepared data is missing. Run data preparation first.")
    frame = pd.read_csv(path, usecols=["series_id", "item_id", "store_id", "state_id", "cat_id", "dept_id"])
    rows = frame.drop_duplicates("series_id").head(limit).to_dict(orient="records")
    return {"count": len(rows), "products": rows}
