from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "household_power_consumption.txt"

def load_series(freq="D", source_path=None):
    """Load and resample Global_active_power to a regular time series."""
    path = Path(source_path) if source_path else DATA
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python download_data.py")
    df = pd.read_csv(
        path, sep=";", usecols=["Date", "Time", "Global_active_power"],
        na_values=["?"], low_memory=False
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    power = pd.to_numeric(df["Global_active_power"], errors="coerce")
    s = pd.Series(power.values, index=dt, name="power_kw").dropna()
    s = s[~s.index.isna()].sort_index()
    s = s.resample(freq).mean().interpolate("time").dropna()
    s.name = "power_kw"
    if s.empty:
        raise ValueError("No valid energy readings were found in the selected dataset.")
    return s

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan}
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_%": float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-6))) * 100)
    }

def _future_index(series, horizon):
    freq = series.index.freq or pd.infer_freq(series.index)
    if freq is None:
        freq = "D"
    return pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]

def _fit_stat_model(name, train, horizon, period):
    if name == "ARIMA":
        model = ARIMA(train, order=(2, 1, 2)).fit()
    else:
        seasonal_period = period if len(train) >= 2 * period else 1
        seasonal = (1, 1, 1, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0)
        model = SARIMAX(
            train, order=(1, 1, 1), seasonal_order=seasonal,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    return model.get_forecast(steps=horizon)

def _prophet_forecast(train, horizon, future_dates):
    from prophet import Prophet
    frame = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame)
    future = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)
    return (
        pd.Series(forecast["yhat"].to_numpy(), index=future_dates),
        pd.Series(forecast["yhat_lower"].to_numpy(), index=future_dates),
        pd.Series(forecast["yhat_upper"].to_numpy(), index=future_dates)
    )

def _lstm_forecast(train, horizon, future_dates, lookback=14, epochs=12):
    """Optional TensorFlow LSTM; imports TensorFlow only when requested."""
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    values = train.to_numpy(dtype="float32").reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).ravel()
    lookback = min(lookback, max(3, len(scaled) // 5))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    X = np.asarray(X).reshape(-1, lookback, 1)
    y = np.asarray(y)
    model = Sequential([LSTM(32, input_shape=(lookback, 1)), Dense(16, activation="relu"), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    window = scaled[-lookback:].copy()
    out = []
    for _ in range(horizon):
        pred = float(model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0])
        out.append(pred)
        window = np.append(window[1:], pred)
    predictions = scaler.inverse_transform(np.asarray(out).reshape(-1, 1)).ravel()
    return pd.Series(predictions, index=future_dates)

def fit_forecasts(series, horizon=30, include_prophet=False, include_lstm=False):
    """Chronological holdout evaluation, then refit on full history for future predictions."""
    series = series.astype(float).dropna().sort_index()
    if len(series) < 40:
        raise ValueError("Need at least 40 observations. Choose a coarser frequency or provide more data.")
    test_n = min(max(7, int(horizon)), max(7, int(len(series) * 0.2)))
    test_n = min(test_n, len(series) - 20)
    train, test = series.iloc[:-test_n], series.iloc[-test_n:]
    freq = str(series.index.freqstr or pd.infer_freq(series.index) or "D").upper()
    period = 52 if freq.startswith("W") else (12 if freq.startswith("M") else 7)
    future_dates_test = test.index
    predictions, intervals, errors = {}, {}, {}

    predictions["Naive"] = pd.Series(train.iloc[-1], index=test.index)
    seasonal_period = period if len(train) >= 2 * period else 1
    predictions["Seasonal Naive"] = pd.Series(
        [train.iloc[-seasonal_period + i % seasonal_period] for i in range(len(test))],
        index=test.index
    )

    for model_name in ["ARIMA", "SARIMA"]:
        try:
            result = _fit_stat_model(model_name, train, len(test), period)
            predictions[model_name] = pd.Series(np.asarray(result.predicted_mean), index=test.index)
            ci = result.conf_int()
            intervals[model_name] = (
                pd.Series(np.asarray(ci.iloc[:, 0]), index=test.index),
                pd.Series(np.asarray(ci.iloc[:, 1]), index=test.index)
            )
        except Exception as exc:
            errors[model_name] = str(exc)

    if include_prophet:
        try:
            pred, low, high = _prophet_forecast(train, len(test), future_dates_test)
            predictions["Prophet"] = pred
            intervals["Prophet"] = (low, high)
        except Exception as exc:
            errors["Prophet"] = f"{exc}. Install with: pip install prophet"

    if include_lstm:
        try:
            predictions["LSTM"] = _lstm_forecast(train, len(test), future_dates_test)
        except Exception as exc:
            errors["LSTM"] = f"{exc}. Install TensorFlow using requirements-advanced.txt"

    scores = {name: calculate_metrics(test, pred) for name, pred in predictions.items()}
    best = min(scores, key=lambda name: scores[name]["MAE"])
    future_dates = _future_index(series, horizon)
    future, lower, upper = None, None, None

    if best in ["ARIMA", "SARIMA"]:
        result = _fit_stat_model(best, series, horizon, period)
        future = pd.Series(np.asarray(result.predicted_mean), index=future_dates)
        ci = result.conf_int()
        lower = pd.Series(np.asarray(ci.iloc[:, 0]), index=future_dates)
        upper = pd.Series(np.asarray(ci.iloc[:, 1]), index=future_dates)
    elif best == "Prophet":
        future, lower, upper = _prophet_forecast(series, horizon, future_dates)
    elif best == "LSTM":
        future = _lstm_forecast(series, horizon, future_dates)
    elif best == "Seasonal Naive":
        p = period if len(series) >= period else 1
        future = pd.Series([series.iloc[-p + i % p] for i in range(horizon)], index=future_dates)
    else:
        future = pd.Series(series.iloc[-1], index=future_dates)

    return {
        "train": train, "test": test, "predictions": predictions,
        "scores": scores, "best": best, "future": future,
        "lower": lower, "upper": upper, "intervals": intervals, "errors": errors
    }

def detect_anomalies(series, contamination=0.02):
    """Isolation Forest anomaly flags on level and rolling variability."""
    frame = pd.DataFrame({"value": series.astype(float)})
    frame["rolling_mean"] = frame["value"].rolling(7, min_periods=2).mean()
    frame["rolling_std"] = frame["value"].rolling(7, min_periods=2).std().fillna(0)
    frame = frame.dropna()
    model = IsolationForest(contamination=float(contamination), random_state=42)
    frame["anomaly"] = model.fit_predict(frame[["value", "rolling_mean", "rolling_std"]]) == -1
    return frame

def estimate_cost(forecast, tariff):
    """Approximate cost using mean active power in kW times period hours and tariff."""
    idx = forecast.index
    if len(idx) == 0:
        return pd.DataFrame(columns=["forecast_power_kw", "estimated_kwh", "estimated_cost"])
    if len(idx) > 1:
        deltas = pd.Series(idx).diff().dropna()
        hours = float(deltas.median().total_seconds() / 3600)
    else:
        hours = 24.0
    frame = pd.DataFrame(index=idx)
    frame["forecast_power_kw"] = forecast.clip(lower=0)
    frame["estimated_kwh"] = frame["forecast_power_kw"] * hours
    frame["estimated_cost"] = frame["estimated_kwh"] * float(tariff)
    return frame
