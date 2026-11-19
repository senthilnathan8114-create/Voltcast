import json
from forecasting import load_series, fit_forecasts, ROOT

series = load_series("D")
result = fit_forecasts(series, horizon=30)
(ROOT / "reports").mkdir(exist_ok=True)
report = {
    "best_model_by_MAE": result["best"],
    "metrics": result["scores"],
    "model_errors": result["errors"],
    "frequency": "Daily",
    "horizon": 30
}
(ROOT / "reports" / "model_metrics.json").write_text(json.dumps(report, indent=2))
forecast = result["future"].rename("forecast").to_frame()
if result["lower"] is not None:
    forecast["lower_bound"] = result["lower"]
    forecast["upper_bound"] = result["upper"]
forecast.to_csv(ROOT / "reports" / "future_forecast.csv")
print("Best model:", result["best"])
print(json.dumps(result["scores"], indent=2))
print("Saved reports/model_metrics.json and reports/future_forecast.csv")
