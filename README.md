# Energy Intelligence: Time Series Forecasting & Anomaly Detection

An enhanced end-to-end data science project using the UCI Individual Household Electric Power Consumption dataset.

## Features

- Daily, weekly and monthly aggregation
- Historical KPIs, data quality checks and rolling statistics
- Seasonal decomposition and Augmented Dickey-Fuller stationarity test
- Chronological model comparison: Naive, Seasonal Naive, ARIMA and SARIMA
- Optional Prophet and LSTM model extensions
- Forecast confidence intervals where supported by the selected model
- Isolation Forest anomaly detection
- Optional weather CSV correlation analysis
- Electricity cost estimate using an editable tariff
- Smart-meter CSV upload and inspection
- Downloadable forecast, anomaly and cost reports

## Dataset

UCI Individual Household Electric Power Consumption:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains 2,075,259 one-minute observations from one household in Sceaux, France, from December 2006 to November 2010. It is household-level historical data, not a representative national or Chennai grid-demand dataset. Cite Hebrail & Berard (2006), DOI: 10.24432/C58K54. License: CC BY 4.0.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and run `.venv\\Scripts\\activate.bat`, or call `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`.

## Optional Prophet and LSTM

The core application does not require TensorFlow or Prophet. To install both optional packages:

```powershell
pip install -r requirements-advanced.txt
```

Prophet and LSTM can be selected from the sidebar. LSTM training is slower and results depend on training settings and data frequency.

## Train and export a baseline model report

```powershell
python train_model.py
```

Outputs are saved under `reports/`.

## Weather CSV format

Upload a CSV with a date/timestamp column and numeric weather columns, for example:

```csv
date,temperature_c,humidity_percent,precipitation_mm
2006-12-16,8.2,75,0.0
2006-12-17,7.8,81,1.2
```

The app aligns it to the selected time frequency and displays correlations. Correlation is descriptive, not causal, and the weather data must cover the same date range as the energy dataset.

## Smart-meter CSV format

Upload a CSV with a timestamp column and a numeric power or energy column. The app validates and visualizes it. This is file-based inspection, not an automatic live IoT connection.

## Methodology and limitations

- Data is split chronologically; random shuffling is not used.
- Model selection is based on holdout MAE. Always compare against naive baselines.
- Forecast intervals are model-based for ARIMA/SARIMA and Prophet. LSTM currently provides point forecasts only.
- Cost estimation uses average kW multiplied by the number of hours represented by each forecast period, then by the user-entered tariff. It is an approximation and does not encode tiered tariffs, taxes or fixed charges.
- Isolation Forest flags unusual statistical patterns; an anomaly is not automatically a fault or fraud event.
- Weather analysis requires a separate matching weather CSV. Weather is not used as an exogenous predictor in the current ARIMA/SARIMA implementation.
- The smart-meter tab does not stream live data. A production IoT version needs a device gateway, authenticated ingestion and a time-series store.
- Prophet and LSTM are optional extensions and should be evaluated using rolling-origin validation before research claims.
- This project is an educational research prototype, not an operational electricity-grid forecasting system.

## Project structure

```text
time_series_forecasting_advanced/
├── app.py
├── forecasting.py
├── download_data.py
├── train_model.py
├── requirements.txt
├── requirements-advanced.txt
├── README.md
├── data/raw/
├── models/
└── reports/
```
