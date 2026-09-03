from _bootstrap import ROOT

import argparse
from pathlib import Path

import pandas as pd

from src.config import settings
from src.inventory.decision_engine import inventory_decision
from src.utils.paths import ensure_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-file", type=Path, help="Optional CSV with series_id,current_inventory,lead_time,service_level,holding_cost,stockout_cost")
    args = parser.parse_args()

    forecast_files = sorted(settings.predictions_dir.glob("forecast_*.csv"))
    if not forecast_files:
        raise FileNotFoundError("No generated forecast CSVs found. Generate at least one forecast first.")
    inventory_lookup = {}
    if args.inventory_file:
        inventory = pd.read_csv(args.inventory_file)
        inventory_lookup = {str(row.series_id): row for row in inventory.itertuples(index=False)}

    rows = []
    for path in forecast_files:
        forecast = pd.read_csv(path)
        series_id = str(forecast["series_id"].iloc[0])
        row = inventory_lookup.get(series_id)
        current_inventory = float(getattr(row, "current_inventory", 0.0)) if row is not None else 0.0
        lead_time = int(getattr(row, "lead_time", settings.default_lead_time_days)) if row is not None else settings.default_lead_time_days
        service_level = float(getattr(row, "service_level", settings.default_service_level)) if row is not None else settings.default_service_level
        holding_cost = float(getattr(row, "holding_cost", 1.0)) if row is not None else 1.0
        stockout_cost = float(getattr(row, "stockout_cost", 5.0)) if row is not None else 5.0
        decision = inventory_decision(
            forecast["forecast"].tolist(), forecast["lower_bound"].tolist(), forecast["upper_bound"].tolist(),
            current_inventory, lead_time, service_level, holding_cost, stockout_cost,
        )
        rows.append({
            "series_id": series_id,
            "forecast_28d": float(forecast["forecast"].sum()),
            "current_inventory": current_inventory,
            **{k: decision[k] for k in ["safety_stock", "reorder_point", "recommended_order_quantity", "stockout_risk", "overstock_risk", "priority_score", "recommendation"]},
        })
    output_frame = pd.DataFrame(rows).sort_values("priority_score", ascending=False)
    ensure_directories(settings.predictions_dir)
    output = settings.predictions_dir / "inventory_recommendations.csv"
    output_frame.to_csv(output, index=False)
    print(output_frame.to_string(index=False))
    print(f"Saved recommendations to {output}")


if __name__ == "__main__":
    main()
