from _bootstrap import ROOT

from src.config import settings
from src.data.loader import load_m5_data
from src.data.validator import validate_m5_data


def main():
    settings.validate()
    data = load_m5_data(data_mode=settings.data_mode, max_series=settings.max_series)
    report = validate_m5_data(data["sales"], data["calendar"], data["prices"])
    print("M5 data validation")
    print("=" * 60)
    for name, exists in report["expected_files"].items():
        print(f"{name:32} {'FOUND' if exists else 'MISSING'}")
    print(f"Sales shape: {report['sales_shape']} | memory: {report['sales_memory_mb']} MB")
    print(f"Calendar shape: {report['calendar_shape']} | memory: {report['calendar_memory_mb']} MB")
    print(f"Prices shape: {report['prices_shape']} | memory: {report['prices_memory_mb']} MB")
    print(f"Missing values — sales/calendar/prices: {report['sales_missing_values']}/{report['calendar_missing_values']}/{report['prices_missing_values']}")
    print(f"Date range: {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"Series: {report['number_of_series']} | stores: {report['number_of_stores']} | products: {report['number_of_products']}")
    if report["calendar_missing_dates"]:
        print(f"Warning: calendar has {report['calendar_missing_dates']} missing date(s).")


if __name__ == "__main__":
    main()
