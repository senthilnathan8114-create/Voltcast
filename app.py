import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from forecasting import (
    load_series, fit_forecasts, detect_anomalies, estimate_cost
)

# ------------------------------------------------------------
# PAGE + VISUAL SYSTEM
# ------------------------------------------------------------
st.set_page_config(
    page_title="Voltcast | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BG = "#0B1020"
PANEL = "#121A2B"
PANEL_2 = "#172238"
TEXT = "#E8EEF9"
MUTED = "#93A4BE"
CYAN = "#42D6C5"
BLUE = "#6EA8FE"
PURPLE = "#B69CFF"
ORANGE = "#FFB86B"
RED = "#FF718B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --bg: {BG}; --panel: {PANEL}; --panel2: {PANEL_2};
  --text: {TEXT}; --muted: {MUTED}; --cyan: {CYAN};
}}
html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(ellipse at 8% 0%, rgba(66,214,197,.10), transparent 28%),
    radial-gradient(ellipse at 95% 10%, rgba(110,168,254,.10), transparent 25%),
    var(--bg);
  color: var(--text);
}}
[data-testid="stHeader"] {{ background: rgba(11,16,32,.82); }}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #111A2C 0%, #0D1424 100%);
  border-right: 1px solid rgba(147,164,190,.14);
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}
.block-container {{ padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1500px; }}
h1,h2,h3 {{ font-family: 'Space Grotesk', sans-serif !important; color: {TEXT}; }}
p, label, .stCaption {{ color: {MUTED}; }}
.hero {{
  padding: 1.65rem 1.8rem;
  border: 1px solid rgba(66,214,197,.22);
  border-radius: 24px;
  background: linear-gradient(115deg, rgba(24,43,62,.96), rgba(18,26,43,.88) 55%, rgba(40,35,75,.75));
  box-shadow: 0 18px 60px rgba(0,0,0,.22);
  margin-bottom: 1.35rem;
  position: relative; overflow: hidden;
}}
.hero:after {{
  content:""; position:absolute; width:260px; height:260px; border-radius:50%;
  right:-75px; top:-130px; background:rgba(66,214,197,.12); filter:blur(4px);
}}
.eyebrow {{
  color:{CYAN}; font-size:.75rem; font-weight:800; letter-spacing:.18em;
  text-transform:uppercase; margin-bottom:.5rem;
}}
.hero-title {{ font-family:'Space Grotesk',sans-serif; font-size:2.25rem; line-height:1.1; font-weight:700; color:{TEXT}; }}
.hero-sub {{ color:{MUTED}; margin-top:.65rem; font-size:1rem; max-width:850px; }}
.pill {{
  display:inline-block; border:1px solid rgba(66,214,197,.3);
  background:rgba(66,214,197,.10); color:{CYAN}; padding:.35rem .7rem;
  border-radius:999px; font-size:.75rem; font-weight:700; margin-right:.35rem;
}}
.metric-card {{
  background:linear-gradient(145deg, rgba(23,34,56,.98), rgba(18,26,43,.98));
  border:1px solid rgba(147,164,190,.15); border-radius:18px;
  padding:1.05rem 1.15rem; min-height:116px;
  box-shadow:0 12px 30px rgba(0,0,0,.12);
}}
.metric-label {{ color:{MUTED}; font-size:.78rem; font-weight:600; }}
.metric-value {{ color:{TEXT}; font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; margin-top:.35rem; }}
.metric-note {{ color:{CYAN}; font-size:.72rem; margin-top:.2rem; }}
.section-kicker {{ color:{CYAN}; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; font-weight:800; }}
div[data-testid="stTabs"] button {{
  background:transparent; color:{MUTED}; border-radius:10px 10px 0 0;
  font-weight:700; padding:.8rem .85rem;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  color:{CYAN}; border-bottom:2px solid {CYAN}; background:rgba(66,214,197,.07);
}}
.stButton > button, .stDownloadButton > button {{
  border-radius:12px; border:1px solid rgba(66,214,197,.35);
  background:linear-gradient(100deg, #1C9F98, #287EAA);
  color:white; font-weight:700; padding:.55rem 1rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color:{CYAN}; color:white; filter:brightness(1.12);
}}
div[data-testid="stDataFrame"] {{
  border:1px solid rgba(147,164,190,.16); border-radius:14px; overflow:hidden;
}}
div[data-testid="stAlert"] {{ border-radius:14px; }}
hr {{ border-color:rgba(147,164,190,.14); }}
.small-muted {{ color:{MUTED}; font-size:.82rem; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="DM Sans"),
    margin=dict(l=20, r=20, t=35, b=20),
    xaxis=dict(showgrid=False, zeroline=False, color=MUTED),
    yaxis=dict(gridcolor="rgba(147,164,190,.13)", zeroline=False, color=MUTED),
    legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)")
)

def styled_line(series, name="Power", color=CYAN, title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", name=name,
        line=dict(color=color, width=2.2),
        fill="tozeroy", fillcolor="rgba(66,214,197,.055)"
    ))
    fig.update_layout(**PLOT_LAYOUT, title=title, height=390,
                      yaxis_title="Mean active power (kW)")
    return fig

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="padding:.5rem 0 1rem"><div class="eyebrow">VOLTCAST / CONTROL ROOM</div>'
        f'<div style="font-family:Space Grotesk;font-size:1.55rem;font-weight:700;color:{TEXT}">⚡ Voltcast</div>'
        f'<div class="small-muted">Energy intelligence platform</div></div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("#### Forecast configuration")
    freq_label = st.selectbox("Aggregation interval", ["Daily", "Weekly", "Monthly"], index=0)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq_label]
    horizon = st.slider("Forecast horizon", 7, 90, 30, help="Number of future periods.")
    st.markdown("#### Model extensions")
    use_prophet = st.checkbox("Prophet", help="Requires the optional Prophet dependency.")
    use_lstm = st.checkbox("LSTM neural network", help="Requires TensorFlow; training can take longer.")
    st.markdown("#### Business assumptions")
    tariff = st.number_input("Electricity tariff (₹ / kWh)", min_value=0.0, value=8.0, step=0.5)
    contamination = st.slider("Anomaly sensitivity", 0.01, 0.10, 0.02, 0.01)
    st.divider()
    st.caption("Source: UCI Individual Household Electric Power Consumption, 2006–2010.")
    st.caption("Historical single-household data • Educational research prototype")

# ------------------------------------------------------------
# LOAD SERIES
# ------------------------------------------------------------
try:
    with st.spinner("Preparing the energy time series..."):
        s = load_series(freq)
except Exception as exc:
    st.markdown('<div class="hero"><div class="eyebrow">DATA CONNECTION</div><div class="hero-title">Your energy workspace is ready</div></div>', unsafe_allow_html=True)
    st.error(f"Could not load the dataset: {exc}")
    st.info("Run `python download_data.py` in the project folder, then refresh this page.")
    st.stop()

if len(s) < 40:
    st.error("The selected interval has too few observations for reliable model evaluation.")
    st.stop()

# ------------------------------------------------------------
# HERO + KPI ROW
# ------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">TIME SERIES ANALYTICS • FORECASTING STUDIO</div>
  <div class="hero-title">Energy, made predictable<span style="color:{CYAN}">.</span></div>
  <div class="hero-sub">Explore consumption behaviour, uncover unusual patterns and forecast future household energy demand — all in one workspace.</div>
  <div style="margin-top:1rem">
    <span class="pill">LIVE ANALYTICS</span><span class="pill">ML MODEL LAB</span><span class="pill">ANOMALY INTELLIGENCE</span>
  </div>
</div>
""", unsafe_allow_html=True)

k = st.columns(4)
with k[0]: metric_card("OBSERVATIONS", f"{len(s):,}", "Aggregated time periods")
with k[1]: metric_card("AVERAGE POWER", f"{s.mean():.3f} kW", "Mean active power")
with k[2]: metric_card("PEAK POWER", f"{s.max():.3f} kW", "Highest period average")
with k[3]: metric_card("DATA WINDOW", f"{s.index.min():%Y}–{s.index.max():%Y}", freq_label + " resolution")

st.markdown("<br>", unsafe_allow_html=True)

tabs = st.tabs([
    "◈ Overview", "⌁ Patterns", "◎ Stationarity", "✦ Forecast Lab",
    "⚑ Anomalies", "☁ Weather", "₹ Cost Studio", "⌁ Meter Upload"
])

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown('<div class="section-kicker">01 / Consumption profile</div>', unsafe_allow_html=True)
        st.plotly_chart(styled_line(s, "Observed power", CYAN, "Historical energy consumption"),
                        width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-kicker">02 / At a glance</div>', unsafe_allow_html=True)
        st.markdown("#### Descriptive statistics")
        stats = s.describe().rename("Power (kW)").to_frame()
        st.dataframe(stats, width="stretch")
        st.markdown("#### Data integrity")
        st.write(f"**Missing values:** {int(s.isna().sum())}")
        st.write(f"**Duplicate timestamps:** {int(s.index.duplicated().sum())}")
        st.write(f"**Negative values:** {int((s < 0).sum())}")
    st.markdown('<div class="section-kicker">03 / Distribution</div>', unsafe_allow_html=True)
    hist = px.histogram(x=s.values, nbins=40, labels={"x": "Mean active power (kW)"}, title="Consumption distribution")
    hist.update_layout(**PLOT_LAYOUT, height=300, showlegend=False, bargap=.06)
    hist.update_traces(marker_color=BLUE, opacity=.82)
    st.plotly_chart(hist, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------
# PATTERNS
# ------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-kicker">Trend decomposition</div>', unsafe_allow_html=True)
    window = max(3, min(30, len(s) // 10))
    roll = s.rolling(window).mean()
    std = s.rolling(window).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Observed", line=dict(color=MUTED, width=1)))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name=f"{window}-period rolling mean",
                             line=dict(color=CYAN, width=2.5)))
    fig.add_trace(go.Scatter(x=roll.index, y=(roll+std).values, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=roll.index, y=(roll-std).values, fill="tonexty",
                             fillcolor="rgba(66,214,197,.10)", line=dict(width=0),
                             name="±1 standard deviation", hoverinfo="skip"))
    fig.update_layout(**PLOT_LAYOUT, height=390, title="Rolling mean and volatility")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    period = {"D": 7, "W": 52, "MS": 12}[freq]
    if len(s) >= 2 * period:
        try:
            dec = seasonal_decompose(s, model="additive", period=period)
            decomp = go.Figure()
            for label, values, color in [
                ("Observed", dec.observed, CYAN),
                ("Trend", dec.trend, BLUE),
                ("Seasonal", dec.seasonal, PURPLE),
                ("Residual", dec.resid, ORANGE)
            ]:
                decomp.add_trace(go.Scatter(x=values.index, y=values.values, name=label,
                                            line=dict(color=color, width=1.6)))
            decomp.update_layout(**PLOT_LAYOUT, height=430, title="Seasonal decomposition")
            st.plotly_chart(decomp, width="stretch", config={"displayModeBar": False})
        except Exception as exc:
            st.warning(f"Decomposition unavailable: {exc}")
    else:
        st.info("Select a longer time span or coarser aggregation to enable seasonal decomposition.")

# ------------------------------------------------------------
# STATIONARITY
# ------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-kicker">Statistical diagnostics</div>', unsafe_allow_html=True)
    if s.nunique() < 2:
        st.warning("ADF test requires variation in the series.")
    else:
        try:
            adf = adfuller(s.dropna(), autolag="AIC", result_object=False)
            diff = s.diff().dropna()
            adf_diff = adfuller(diff, autolag="AIC", result_object=False)
            a, b, c = st.columns(3)
            with a: metric_card("ORIGINAL ADF P-VALUE", f"{adf[1]:.5f}", "Below 0.05 suggests stationarity")
            with b: metric_card("DIFFERENCED P-VALUE", f"{adf_diff[1]:.5f}", "After first-order differencing")
            with c: metric_card("ADF STATISTIC", f"{adf[0]:.3f}", "Original series")
            if adf[1] < .05:
                st.success("The original series appears stationary under the ADF test.")
            else:
                st.warning("The original series appears non-stationary; differencing may help.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=diff.index, y=diff.values, name="First difference",
                                     line=dict(color=PURPLE, width=1.4)))
            fig.update_layout(**PLOT_LAYOUT, height=330, title="First-order differencing")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        except Exception as exc:
            st.error(f"ADF test failed: {exc}")

# ------------------------------------------------------------
# FORECAST LAB
# ------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-kicker">Machine learning model arena</div>', unsafe_allow_html=True)
    st.markdown("#### Compare forecasting approaches")
    st.caption("Time-ordered holdout evaluation. Naive baselines are included to ensure complex models earn their place.")
    if st.button("⚡ Run model comparison", type="primary", key="run_forecast"):
        with st.spinner("Training and evaluating models..."):
            try:
                st.session_state["forecast_result"] = fit_forecasts(
                    s, horizon, include_prophet=use_prophet, include_lstm=use_lstm
                )
            except Exception as exc:
                st.error(f"Forecast training failed: {exc}")
    result = st.session_state.get("forecast_result")
    if result:
        st.success(f"Best validation model by MAE: {result['best']}")
        score_df = pd.DataFrame(result["scores"]).T.sort_values("MAE")
        st.dataframe(score_df.style.format("{:.3f}"), width="stretch")
        if result["errors"]:
            with st.expander("Optional models unavailable / training errors"):
                for name, err in result["errors"].items():
                    st.warning(f"{name}: {err}")
        compare = go.Figure()
        compare.add_trace(go.Scatter(x=result["test"].index, y=result["test"].values,
                                     name="Actual", line=dict(color=TEXT, width=3)))
        colors = [CYAN, BLUE, PURPLE, ORANGE, RED]
        for i, (name, pred) in enumerate(result["predictions"].items()):
            compare.add_trace(go.Scatter(x=pred.index, y=pred.values, name=name,
                                         line=dict(color=colors[i % len(colors)], width=1.8)))
        compare.update_layout(**PLOT_LAYOUT, height=390, title="Holdout: actual vs predicted")
        st.plotly_chart(compare, width="stretch", config={"displayModeBar": False})
        future_fig = go.Figure()
        hist = s.iloc[-min(180, len(s)):]
        future_fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History",
                                        line=dict(color=BLUE, width=2)))
        future_fig.add_trace(go.Scatter(x=result["future"].index, y=result["future"].values,
                                        name=f"Forecast · {result['best']}",
                                        line=dict(color=CYAN, width=2.8)))
        if result["lower"] is not None and result["upper"] is not None:
            future_fig.add_trace(go.Scatter(x=result["future"].index, y=result["upper"].values,
                                            line=dict(width=0), showlegend=False, hoverinfo="skip"))
            future_fig.add_trace(go.Scatter(x=result["future"].index, y=result["lower"].values,
                                            fill="tonexty", fillcolor="rgba(66,214,197,.15)",
                                            line=dict(width=0), name="Prediction interval"))
        future_fig.update_layout(**PLOT_LAYOUT, height=410, title="Future energy outlook")
        st.plotly_chart(future_fig, width="stretch", config={"displayModeBar": False})
        out = pd.DataFrame({"forecast": result["future"]})
        if result["lower"] is not None:
            out["lower_bound"] = result["lower"]
            out["upper_bound"] = result["upper"]
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("↓ Export forecast CSV", out.to_csv().encode(), "energy_forecast.csv", "text/csv")
        with c2:
            report = {"best_model": result["best"], "metrics": result["scores"], "errors": result["errors"]}
            st.download_button("↓ Export model report", pd.Series(report).to_json(indent=2),
                               "model_evaluation.json", "application/json")
    else:
        st.info("Choose optional models in the sidebar and run the model comparison.")

# ------------------------------------------------------------
# ANOMALIES
# ------------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="section-kicker">Outlier intelligence</div>', unsafe_allow_html=True)
    st.caption("Isolation Forest identifies unusual statistical patterns. Review flagged periods before interpreting them as faults.")
    try:
        anomaly = detect_anomalies(s, contamination)
        flagged = anomaly[anomaly["anomaly"]]
        a, b = st.columns(2)
        with a: metric_card("FLAGGED PERIODS", f"{len(flagged):,}", "Potential anomalies")
        with b: metric_card("ANOMALY RATE", f"{len(flagged)/max(1,len(anomaly))*100:.2f}%", "Of analyzed observations")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anomaly.index, y=anomaly["value"], name="Consumption",
                                 mode="lines", line=dict(color=BLUE, width=1.5)))
        if not flagged.empty:
            fig.add_trace(go.Scatter(x=flagged.index, y=flagged["value"], name="Potential anomaly",
                                     mode="markers", marker=dict(color=RED, size=7, symbol="x")))
        fig.update_layout(**PLOT_LAYOUT, height=390, title="Consumption anomalies")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.dataframe(flagged.tail(100), width="stretch")
        st.download_button("↓ Download anomaly report", anomaly.to_csv().encode(),
                           "energy_anomalies.csv", "text/csv")
    except Exception as exc:
        st.error(f"Anomaly detection failed: {exc}")

# ------------------------------------------------------------
# WEATHER
# ------------------------------------------------------------
with tabs[5]:
    st.markdown('<div class="section-kicker">External signal explorer</div>', unsafe_allow_html=True)
    st.write("Upload weather observations covering the same period as the energy data.")
    weather_file = st.file_uploader("Choose weather CSV", type=["csv"], key="weather_upload")
    if weather_file:
        try:
            weather = pd.read_csv(weather_file)
            date_col = st.selectbox("Weather date column", weather.columns, key="weather_date")
            weather[date_col] = pd.to_datetime(weather[date_col], errors="coerce")
            weather = weather.dropna(subset=[date_col]).set_index(date_col).sort_index()
            numeric = weather.select_dtypes(include=np.number).columns.tolist()
            if numeric:
                aligned = pd.concat([s.rename("energy_power_kw"),
                                     weather[numeric].resample(freq).mean()], axis=1).dropna()
                st.metric("Matched periods", f"{len(aligned):,}")
                st.dataframe(aligned.corr(numeric_only=True), width="stretch")
                variable = st.selectbox("Explore weather variable", numeric)
                fig = px.scatter(aligned, x=variable, y="energy_power_kw",
                                 trendline="ols", title=f"Energy vs {variable}")
                fig.update_layout(**PLOT_LAYOUT, height=370)
                fig.update_traces(marker_color=CYAN, marker_opacity=.65)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("Correlation is descriptive; it does not establish causation.")
            else:
                st.warning("No numeric weather columns were found.")
        except Exception as exc:
            st.error(f"Weather file processing failed: {exc}")
    else:
        st.info("Weather analysis is optional and works with a matching weather CSV. No API key is required.")

# ------------------------------------------------------------
# COST STUDIO
# ------------------------------------------------------------
with tabs[6]:
    st.markdown('<div class="section-kicker">Business impact estimator</div>', unsafe_allow_html=True)
    result = st.session_state.get("forecast_result")
    if result:
        cost = estimate_cost(result["future"], tariff)
        a, b = st.columns(2)
        with a: metric_card("ESTIMATED ENERGY", f"{cost['estimated_kwh'].sum():,.1f} kWh", "Across forecast horizon")
        with b: metric_card("ESTIMATED COST", f"₹{cost['estimated_cost'].sum():,.2f}", f"At ₹{tariff:.2f} per kWh")
        fig = go.Figure(go.Bar(x=cost.index, y=cost["estimated_cost"],
                               marker=dict(color=CYAN, line=dict(width=0))))
        fig.update_layout(**PLOT_LAYOUT, height=360, title="Estimated cost by forecast period",
                          yaxis_title="Estimated cost (₹)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.dataframe(cost, width="stretch")
        st.download_button("↓ Export cost estimate", cost.to_csv().encode(),
                           "energy_cost_estimate.csv", "text/csv")
        st.caption("Approximation based on average kW × period hours × tariff. Does not model slab rates, taxes or fixed charges.")
    else:
        st.info("Run a forecast in the Forecast Lab first to estimate future cost.")

# ------------------------------------------------------------
# METER UPLOAD
# ------------------------------------------------------------
with tabs[7]:
    st.markdown('<div class="section-kicker">Bring your own meter data</div>', unsafe_allow_html=True)
    st.caption("Upload a smart-meter CSV for inspection. This is file-based analysis, not a live IoT connection.")
    meter_file = st.file_uploader("Choose meter CSV", type=["csv"], key="meter_upload")
    if meter_file:
        try:
            meter = pd.read_csv(meter_file)
            date_col = st.selectbox("Timestamp field", meter.columns, key="meter_timestamp")
            numeric_cols = meter.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric reading column found.")
            else:
                value_col = st.selectbox("Power / energy field", numeric_cols, key="meter_value")
                meter[date_col] = pd.to_datetime(meter[date_col], errors="coerce")
                readings = meter.dropna(subset=[date_col]).set_index(date_col)[value_col].dropna().sort_index()
                metric_card("UPLOADED READINGS", f"{len(readings):,}", "Validated numeric observations")
                fig = go.Figure(go.Scatter(x=readings.index, y=readings.values, mode="lines",
                                           line=dict(color=PURPLE, width=1.8)))
                fig.update_layout(**PLOT_LAYOUT, height=360, title="Uploaded meter readings")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.dataframe(readings.describe().to_frame(value_col), width="stretch")
        except Exception as exc:
            st.error(f"Meter file processing failed: {exc}")
    else:
        st.info("Connect a physical meter through a secure gateway or time-series database in a future deployment.")

st.markdown(
    f'<div style="margin-top:2rem;padding:1rem 0;border-top:1px solid rgba(147,164,190,.15);'
    f'color:{MUTED};font-size:.78rem;display:flex;justify-content:space-between;flex-wrap:wrap">'
    f'<span>VOLTCAST · ENERGY INTELLIGENCE</span>'
    f'<span>UCI household data · Historical single-household research prototype</span></div>',
    unsafe_allow_html=True
)
