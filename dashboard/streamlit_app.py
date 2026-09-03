from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import settings
from src.inventory.decision_engine import inventory_decision

st.set_page_config(page_title="Retail Forecasting & Inventory", layout="wide")
st.title("Retail Demand Forecasting & Inventory Decision Engine")
st.caption("Results appear after the local M5 pipeline has been executed.")

prepared_path = settings.processed_dir / "m5_prepared.csv"
predictions_dir = settings.predictions_dir
metrics_path = settings.metrics_dir / "validation_metrics.json"

if not prepared_path.exists():
    st.warning("Prepared data is missing. Run `python scripts/prepare_data.py` first.")
    st.stop()

@st.cache_data
def load_prepared():
    return pd.read_csv(prepared_path, parse_dates=["date"])

frame = load_prepared()
metadata = frame[["series_id", "state_id", "store_id", "cat_id", "dept_id", "item_id"]].drop_duplicates()

st.sidebar.header("SKU / Store Drilldown")
state = st.sidebar.selectbox("State", ["All"] + sorted(metadata["state_id"].astype(str).unique().tolist()))
filtered = metadata if state == "All" else metadata[metadata["state_id"].astype(str) == state]
store = st.sidebar.selectbox("Store", ["All"] + sorted(filtered["store_id"].astype(str).unique().tolist()))
if store != "All": filtered = filtered[filtered["store_id"].astype(str) == store]
category = st.sidebar.selectbox("Category", ["All"] + sorted(filtered["cat_id"].astype(str).unique().tolist()))
if category != "All": filtered = filtered[filtered["cat_id"].astype(str) == category]
department = st.sidebar.selectbox("Department", ["All"] + sorted(filtered["dept_id"].astype(str).unique().tolist()))
if department != "All": filtered = filtered[filtered["dept_id"].astype(str) == department]
series_id = st.sidebar.selectbox("SKU / Store Series", sorted(filtered["series_id"].astype(str).unique().tolist()))

prediction_path = predictions_dir / f"forecast_{series_id}.csv"
st.header("Forecast Overview")
if prediction_path.exists():
    pred = pd.read_csv(prediction_path, parse_dates=["date"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast Demand", f"{pred['forecast'].sum():,.1f}")
    c2.metric("Lower Bound", f"{pred['lower_bound'].sum():,.1f}")
    c3.metric("Upper Bound", f"{pred['upper_bound'].sum():,.1f}")
    st.write(f"Forecast horizon: {len(pred)} days")

    st.subheader("Forecast vs Actual")
    history = frame[frame["series_id"].astype(str) == series_id].sort_values("date").tail(90)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history["date"], history["demand"], label="Actual")
    ax.plot(pred["date"], pred["forecast"], label="Forecast")
    ax.fill_between(pred["date"], pred["lower_bound"], pred["upper_bound"], alpha=0.2, label="Prediction interval")
    ax.set_title("Demand History and Forecast")
    ax.set_ylabel("Units")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    st.header("Inventory Decision")
    current_inventory = st.number_input("Current inventory", min_value=0.0, value=float(max(0.0, pred["forecast"].sum() * 0.5)))
    lead_time = st.number_input("Lead time (days)", min_value=1, max_value=len(pred), value=min(7, len(pred)))
    service = st.selectbox("Service level", [0.90, 0.95, 0.99], index=1)
    holding = st.number_input("Holding cost per unit", min_value=0.0, value=1.0)
    stockout = st.number_input("Stockout cost per unit", min_value=0.0, value=5.0)
    decision = inventory_decision(
        pred["forecast"].tolist(), pred["lower_bound"].tolist(), pred["upper_bound"].tolist(),
        current_inventory, int(lead_time), service, holding, stockout,
    )
    st.json(decision)
else:
    st.info(f"No generated forecast found for {series_id}. Run the forecast script for this series.")

st.header("Risk Ranking")
risk_path = predictions_dir / "inventory_recommendations.csv"
if risk_path.exists():
    ranking = pd.read_csv(risk_path).sort_values("priority_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    st.dataframe(ranking, use_container_width=True)
else:
    st.info("Run the inventory engine to generate the risk ranking table.")

st.header("Model Performance")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
else:
    st.info("Run model training to generate evaluation metrics.")
