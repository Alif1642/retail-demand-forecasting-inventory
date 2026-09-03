from fastapi import FastAPI

from api.routes.backtest import router as backtest_router
from api.routes.forecast import router as forecast_router
from api.routes.health import router as health_router
from api.routes.inventory import router as inventory_router
from api.routes.products import router as products_router

app = FastAPI(
    title="Retail Demand Forecasting & Inventory Decision Engine",
    version="1.0.0",
    description="M5-based demand forecasting and inventory decision support API.",
)

app.include_router(health_router)
app.include_router(products_router)
app.include_router(forecast_router)
app.include_router(inventory_router)
app.include_router(backtest_router)
