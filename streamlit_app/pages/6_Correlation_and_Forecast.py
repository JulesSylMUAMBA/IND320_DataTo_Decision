# pages/6_Correlation_and_Forecast.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    fetch_price_areas,
    fetch_groups,
    fetch_energy_timeseries,
    fetch_openmeteo,
    calculate_sliding_correlation,
    run_sarimax_forecast,
)

st.set_page_config(page_title="Correlation & Forecast", page_icon="📈", layout="wide")

st.title("📈 Correlation & Forecast")
st.caption("Sliding Window Correlation + SARIMAX Forecast using MongoDB (Elhub) and Open-Meteo ERA5.")

# =====================================================================
#  🔵 TAB 1 — SLIDING WINDOW CORRELATION
# =====================================================================
tab_corr, tab_sarimax = st.tabs(["🔵 Sliding Window Correlation", "🟣 SARIMAX Forecast"])

# =====================================================================
# TAB 1 — CORRELATION
# =====================================================================
with tab_corr:
    st.subheader("Sliding Window Correlation")

    # ---------------------------
    # Controls
    # ---------------------------
    colA, colB, colC, colD = st.columns(4)

    with colA:
        area_corr = st.selectbox(
            "Price area",
            fetch_price_areas("production_mba_hour"),
            key="corr_area",
        )

    with colB:
        kind_corr = st.selectbox(
            "Dataset",
            ["Production", "Consumption"],
            key="corr_kind",
        )

        if kind_corr == "Production":
            coll_corr = "production_mba_hour"
            group_field_corr = "production_group"
        else:
            coll_corr = "consumption_mba_hour"
            group_field_corr = "consumption_group"

    with colC:
        groups_corr = fetch_groups(coll_corr, group_field_corr)
        groups_corr = [g for g in groups_corr if g != "*"]

        group_corr = st.selectbox(
            "Group",
            groups_corr,
            key="corr_group",
        )

    with colD:
        meteo_var = st.selectbox(
            "Meteorological variable",
            ["temperature_2m", "precipitation", "wind_speed_10m"],
            key="corr_meteo_var",
        )

    colE, colF = st.columns(2)

    with colE:
        window_corr = st.number_input(
            "Window size (hours)",
            min_value=24,
            max_value=24 * 60,
            value=24 * 7,
            step=24,
            key="corr_window",
        )
    with colF:
        lag_corr = st.number_input(
            "Lag (hours)",
            min_value=-72,
            max_value=72,
            value=0,
            step=1,
            key="corr_lag",
        )

    # Timeframe
    colT1, colT2 = st.columns(2)
    with colT1:
        start_corr = st.date_input("Start date", datetime(2023, 1, 1), key="corr_start")
    with colT2:
        end_corr = st.date_input("End date", datetime(2023, 3, 1), key="corr_end")

    start_dt_corr = datetime.combine(start_corr, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt_corr = datetime.combine(end_corr, datetime.min.time()).replace(tzinfo=timezone.utc)

    # ---------------------------
    # Fetch energy + weather
    # ---------------------------
    st.write("### Data Fetching")

    df_energy = fetch_energy_timeseries(
        coll_name=coll_corr,
        price_area=area_corr,
        group=group_corr,
        start_time_utc=start_dt_corr,
        end_time_utc=end_dt_corr,
        group_field=group_field_corr,
    )

    if df_energy.empty:
        st.error("No energy data found for this interval.")
        st.stop()

    df_weather = fetch_openmeteo(area_corr, start_corr.year)
    df_weather = df_weather.set_index("time").loc[start_dt_corr:end_dt_corr]

    df_join = df_energy.join(df_weather, how="inner")

    if df_join.empty:
        st.error("Failed to merge weather & energy data.")
        st.stop()

    # ---------------------------
    # Compute correlation
    # ---------------------------
    corr_series = calculate_sliding_correlation(
        df_join,
        window_corr,
        "quantity_kwh",
        meteo_var,
        lag_corr,
    )

    fig_corr = px.line(
        corr_series.dropna(),
        title=f"Sliding Correlation ({window_corr}h window) — {group_corr} vs {meteo_var}",
        labels={"value": "Correlation", "index": "Time (UTC)"},
    )
    fig_corr.add_hline(y=0, line_dash="dash", line_color="red")

    st.plotly_chart(fig_corr, use_container_width=True)


# =====================================================================
#  🟣 TAB 2 — SARIMAX
# =====================================================================
with tab_sarimax:

    st.subheader("SARIMAX Forecast")

    # ---------------------------
    # Controls
    # ---------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        area2 = st.selectbox(
            "Price area",
            fetch_price_areas("production_mba_hour"),
            key="sarimax_area",
        )

    with col2:
        kind2 = st.selectbox("Dataset", ["Production", "Consumption"], key="sarimax_kind")
        if kind2 == "Production":
            coll2 = "production_mba_hour"
            group_field2 = "production_group"
        else:
            coll2 = "consumption_mba_hour"
            group_field2 = "consumption_group"

    with col3:
        groups2 = fetch_groups(coll2, group_field2)
        groups2 = [g for g in groups2 if g != "*"]
        group2 = st.selectbox(
            "Group",
            groups2,
            key="sarimax_group",   # IMPORTANT FIX
        )

    colD1, colD2 = st.columns(2)
    with colD1:
        train_start = st.date_input("Training start", datetime(2023, 1, 1), key="sarimax_train_start")
    with colD2:
        train_end = st.date_input("Training end", datetime(2023, 3, 1), key="sarimax_train_end")

    train_start_dt = datetime.combine(train_start, datetime.min.time()).replace(tzinfo=timezone.utc)
    train_end_dt = datetime.combine(train_end, datetime.min.time()).replace(tzinfo=timezone.utc)

    # SARIMAX controls
    colP, colQ, colS = st.columns(3)

    with colP:
        order_p = st.number_input("p", 0, 5, 1)
    with colQ:
        order_d = st.number_input("d", 0, 2, 0)
    with colS:
        order_q = st.number_input("q", 0, 5, 0)

    colSP, colSD, colSQ, colSS = st.columns(4)
    with colSP:
        sp = st.number_input("P", 0, 3, 1)
    with colSD:
        sd = st.number_input("D", 0, 2, 0)
    with colSQ:
        sq = st.number_input("Q", 0, 3, 0)
    with colSS:
        ss = st.number_input("Season length", 1, 48, 24)
    
    st.markdown("""
        **How to read these SARIMAX parameters:**

        - **p, d, q**: non-seasonal ARIMA part  
        - `p` = how many past values (lags) are used  
        - `d` = how many times we difference the series (0 = raw level, 1 = changes)  
        - `q` = how many past errors (shocks) are used  

        - **P, D, Q, s**: seasonal ARIMA part  
        - `P, D, Q` = same idea as p, d, q but for the repeating seasonal pattern  
        - `s` = season length (here, 24 = daily pattern on hourly data)

        For this dashboard, a reasonable starting point is:
        - **(p, d, q) = (1, 0, 0)**  
        - **(P, D, Q, s) = (1, 0, 0, 24)**  

        Then you can experiment with higher values to see how the forecast and confidence intervals react.
        """)


    horizon = st.number_input("Forecast horizon (hours)", 1, 960, 48, key="sarimax_horizon")

    # ---------------------------
    # Fetch energy and weather
    # ---------------------------
    df_e2 = fetch_energy_timeseries(
        coll_name=coll2,
        price_area=area2,
        group=group2,
        start_time_utc=train_start_dt,
        end_time_utc=train_end_dt,
        group_field=group_field2,
    )

    if df_e2.empty:
        st.error("No energy data found for this SARIMAX training period.")
        st.stop()

    df_met2 = fetch_openmeteo(area2, train_start.year)
    df_met2 = df_met2.set_index("time").loc[train_start_dt:train_end_dt]

    df_smx = df_e2.join(df_met2, how="inner")

    if df_smx.empty:
        st.error("Energy & weather could not be aligned.")
        st.stop()

    # Endogenous / Exogenous
    endog = df_smx["quantity_kwh"]
    exog = df_smx[["temperature_2m"]]

    # ---------------------------
    # Run SARIMAX
    # ---------------------------
    forecast_df = run_sarimax_forecast(
        endog_series=endog,
        exog_df=exog,
        periods=horizon,
        order=(order_p, order_d, order_q),
        seasonal_order=(sp, sd, sq, ss),
    )

    if forecast_df.empty:
        st.error("SARIMAX failed.")
        st.stop()

    # ---------------------------
    # Plot forecast
    # ---------------------------
    df_plot = pd.concat(
        [endog.rename("Actual"), forecast_df["mean"].rename("Forecast")],
        axis=1,
    )

    fig_smx = px.line(df_plot, title="SARIMAX Forecast", labels={"index": "Time (UTC)", "value": "kWh"})
    fig_smx.add_trace(px.line(forecast_df["mean_ci_lower"]).data[0])
    fig_smx.add_trace(px.line(forecast_df["mean_ci_upper"]).data[0])

    fig_smx.data[0].name = "Actual"
    fig_smx.data[1].name = "Forecast"
    fig_smx.data[3].name = "Upper 95% CI"
    fig_smx.data[2].name = "Lower 95% CI"
    

    fig_smx.data[0].line.color = "#1f77b4"  # Blue
    fig_smx.data[1].line.color = "#d62728"  # Red
    fig_smx.data[2].line.color = "#2ca02c"  # Green
    fig_smx.data[3].line.color = "#9467bd"  # Purple
    

    st.plotly_chart(fig_smx, use_container_width=True)
