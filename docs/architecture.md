# Architecture

The project is organized as a production-style Python repository rather than a notebook-only analysis.

```mermaid
flowchart TD
    A[M5 Retail Dataset] --> B[Data Validation]
    B --> C[Data Preparation]
    C --> D[Leakage-Safe Feature Engineering]
    D --> E[Seasonal Naive Baseline]
    D --> F[LightGBM Forecasting]
    E --> G[Rolling-Origin Backtesting]
    F --> G
    G --> H[Prediction Intervals]
    H --> I[Inventory Decision Engine]
    I --> J[Stockout / Overstock Risk]
    J --> K[Reorder Recommendation]
    K --> L[FastAPI]
    K --> M[Streamlit Dashboard]
```

`src/` contains reusable production logic. `scripts/` provides command-line workflows, `api/` exposes service endpoints, `dashboard/` provides an interactive view, and `notebooks/` demonstrates each stage by importing production functions.
