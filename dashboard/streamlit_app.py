from __future__ import annotations

from pathlib import Path
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.inventory.decision_engine import inventory_decision


# -------------------------------------------------------------------
# App configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Retail Forecasting & Inventory",
    page_icon="📦",
    layout="wide",
)

DEFAULT_API_URL = (
    "https://retail-demand-forecasting-inventory-1.onrender.com"
)


def get_api_base_url() -> str:
    """Resolve API URL from Streamlit secrets/env with Render fallback."""

    try:
        secret_url = st.secrets.get("API_BASE_URL")
    except Exception:
        secret_url = None

    api_url = (
        secret_url
        or os.getenv("API_BASE_URL")
        or DEFAULT_API_URL
    )

    return str(api_url).rstrip("/")


API_BASE_URL = get_api_base_url()


# -------------------------------------------------------------------
# API helper
# -------------------------------------------------------------------

def api_get(
    path: str,
    params: dict | None = None,
    timeout: int = 90,
) -> dict:
    """Send GET request to FastAPI backend."""

    url = f"{API_BASE_URL}{path}"

    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "User-Agent": "retail-forecasting-streamlit-dashboard"
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)

        raise RuntimeError(
            f"API returned HTTP {exc.code}: {detail}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Unable to connect to forecasting API: {exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "Forecasting API timed out."
        ) from exc


# -------------------------------------------------------------------
# Cached API data
# -------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_products() -> pd.DataFrame:
    payload = api_get("/products")

    products = payload.get("products", [])

    return pd.DataFrame(products)


@st.cache_data(ttl=300)
def load_forecast(
    series_id: str,
    horizon: int = 28,
) -> pd.DataFrame:

    payload = api_get(
        f"/forecast/{quote(series_id, safe='')}",
        {"horizon": horizon},
    )

    forecast = payload.get("forecast", [])

    frame = pd.DataFrame(forecast)

    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])

    return frame


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title(
    "📦 Retail Demand Forecasting & Inventory Decision Engine"
)

st.caption(
    "LightGBM demand forecasts served through a FastAPI backend "
    "with interactive inventory decision support."
)


# -------------------------------------------------------------------
# Backend status
# -------------------------------------------------------------------

try:
    health = api_get("/health")

    status = health.get("status", "unknown")
    data_mode = health.get("data_mode", "unknown")

except RuntimeError as exc:
    st.error(
        "The FastAPI backend is currently unavailable."
    )

    st.code(str(exc))

    st.info(
        "Render Free services may need time to wake after inactivity. "
        "Refresh the page after the backend starts."
    )

    st.stop()


# -------------------------------------------------------------------
# Products
# -------------------------------------------------------------------

try:
    metadata = load_products()

except RuntimeError as exc:
    st.error("Unable to load demo products.")
    st.code(str(exc))
    st.stop()


if metadata.empty:
    st.warning("No demo products are currently available.")
    st.stop()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

st.sidebar.title("Dashboard Controls")

st.sidebar.success(
    f"API: {status.upper()} | Mode: {data_mode}"
)

st.sidebar.markdown("---")

st.sidebar.header("SKU / Store Drilldown")


state_options = [
    "All",
    *sorted(metadata["state_id"].astype(str).unique()),
]

state = st.sidebar.selectbox(
    "State",
    state_options,
)

filtered = metadata.copy()

if state != "All":
    filtered = filtered[
        filtered["state_id"].astype(str) == state
    ]


store_options = [
    "All",
    *sorted(filtered["store_id"].astype(str).unique()),
]

store = st.sidebar.selectbox(
    "Store",
    store_options,
)

if store != "All":
    filtered = filtered[
        filtered["store_id"].astype(str) == store
    ]


category_options = [
    "All",
    *sorted(filtered["cat_id"].astype(str).unique()),
]

category = st.sidebar.selectbox(
    "Category",
    category_options,
)

if category != "All":
    filtered = filtered[
        filtered["cat_id"].astype(str) == category
    ]


department_options = [
    "All",
    *sorted(filtered["dept_id"].astype(str).unique()),
]

department = st.sidebar.selectbox(
    "Department",
    department_options,
)

if department != "All":
    filtered = filtered[
        filtered["dept_id"].astype(str) == department
    ]


series_options = sorted(
    filtered["series_id"].astype(str).unique()
)

series_id = st.sidebar.selectbox(
    "SKU / Store Series",
    series_options,
)


# -------------------------------------------------------------------
# Product information
# -------------------------------------------------------------------

selected = metadata[
    metadata["series_id"].astype(str) == series_id
].iloc[0]


st.subheader("Selected Product")

p1, p2, p3, p4 = st.columns(4)

p1.metric(
    "Item",
    str(selected["item_id"]),
)

p2.metric(
    "Store",
    str(selected["store_id"]),
)

p3.metric(
    "Department",
    str(selected["dept_id"]),
)

p4.metric(
    "State",
    str(selected["state_id"]),
)


# -------------------------------------------------------------------
# Forecast
# -------------------------------------------------------------------

st.header("Demand Forecast")

try:
    pred = load_forecast(
        series_id,
        horizon=28,
    )

except RuntimeError as exc:
    st.error("Unable to retrieve forecast.")
    st.code(str(exc))
    st.stop()


if pred.empty:
    st.warning(
        "No forecast is available for this product."
    )

    st.stop()


forecast_total = pred["forecast"].sum()
lower_total = pred["lower_bound"].sum()
upper_total = pred["upper_bound"].sum()


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "28-Day Forecast",
    f"{forecast_total:,.1f} units",
)

c2.metric(
    "Lower Bound",
    f"{lower_total:,.1f}",
)

c3.metric(
    "Upper Bound",
    f"{upper_total:,.1f}",
)

c4.metric(
    "Forecast Horizon",
    f"{len(pred)} days",
)


# -------------------------------------------------------------------
# Forecast chart
# -------------------------------------------------------------------

st.subheader("Forecast & Prediction Interval")

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    pred["date"],
    pred["forecast"],
    marker="o",
    label="Forecast",
)

ax.fill_between(
    pred["date"],
    pred["lower_bound"],
    pred["upper_bound"],
    alpha=0.2,
    label="Prediction Interval",
)

ax.set_xlabel("Date")
ax.set_ylabel("Demand Units")
ax.set_title(
    f"28-Day Demand Forecast — {series_id}"
)

ax.legend()

fig.autofmt_xdate()

st.pyplot(fig)

with st.expander("View forecast data"):
    st.dataframe(
        pred,
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# Inventory decision engine
# -------------------------------------------------------------------

st.header("Inventory Decision Engine")

i1, i2 = st.columns(2)

with i1:

    default_inventory = float(
        max(
            0.0,
            forecast_total * 0.5,
        )
    )

    current_inventory = st.number_input(
        "Current Inventory",
        min_value=0.0,
        value=default_inventory,
        step=1.0,
    )

    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=1,
        max_value=len(pred),
        value=min(7, len(pred)),
    )

    service = st.selectbox(
        "Service Level",
        [0.90, 0.95, 0.99],
        index=1,
    )


with i2:

    holding = st.number_input(
        "Holding Cost per Unit",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )

    stockout = st.number_input(
        "Stockout Cost per Unit",
        min_value=0.0,
        value=5.0,
        step=0.1,
    )


decision = inventory_decision(
    pred["forecast"].tolist(),
    pred["lower_bound"].tolist(),
    pred["upper_bound"].tolist(),
    current_inventory,
    int(lead_time),
    service,
    holding,
    stockout,
)


st.subheader("Recommended Action")

recommendation = str(
    decision.get(
        "recommendation",
        "N/A",
    )
)

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Recommendation",
    recommendation,
)

d2.metric(
    "Reorder Point",
    f"{decision.get('reorder_point', 0):,.2f}",
)

d3.metric(
    "Recommended Order",
    f"{decision.get('recommended_order_quantity', 0):,.2f}",
)

d4.metric(
    "Priority Score",
    f"{decision.get('priority_score', 0):,.2f}",
)


with st.expander("View complete inventory decision"):
    st.json(decision)


# -------------------------------------------------------------------
# Model performance
# -------------------------------------------------------------------

st.header("Model Performance")

metrics_path = (
    ROOT
    / "results"
    / "metrics"
    / "validation_metrics.json"
)

if metrics_path.exists():

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    st.dataframe(
        pd.DataFrame([metrics]),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Model evaluation metrics are not available "
        "in this deployment."
    )


# -------------------------------------------------------------------
# Architecture
# -------------------------------------------------------------------

st.caption(
    "Powered by LightGBM • FastAPI • Streamlit"
)