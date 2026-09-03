from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.inventory.decision_engine import inventory_decision

router = APIRouter()


@router.post("/inventory/recommendation")
async def inventory_recommendation(request: Request) -> dict:
    try:
        payload = await request.json()
        required = ["forecast", "lower_bound", "upper_bound", "current_inventory"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        return inventory_decision(
            forecast=[float(v) for v in payload["forecast"]],
            lower_bound=[float(v) for v in payload["lower_bound"]],
            upper_bound=[float(v) for v in payload["upper_bound"]],
            current_inventory=float(payload["current_inventory"]),
            lead_time_days=int(payload.get("lead_time", 7)),
            service_level=float(payload.get("service_level", 0.95)),
            holding_cost=float(payload.get("holding_cost", 1.0)),
            stockout_cost=float(payload.get("stockout_cost", 5.0)),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
