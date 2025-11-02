# pages/6_Outliers_Anomalies.py
import streamlit as st
from utils import fetch_openmeteo, CITY_MAP

st.set_page_config(page_title="Outliers & Anomalies", layout="wide")
st.title("New B — Outlier/SPC & Anomaly/LOF")

# --- Read global area from page 2; fallback to local selector if missing ---
area = st.session_state.get("price_area")
if area is None:
    area = st.selectbox("Price area (local)", list(CITY_MAP.keys()), index=4)  # NO5 by default
city = CITY_MAP[area]["city"]

# --- Controls ---
colA, colB = st.columns(2)
with colA:
    year = st.number_input("Year", min_value=2010, max_value=2025, value=2019, step=1)
with colB:
    st.caption(f"Downloading ERA5 for **{city} ({area})**, year **{year}**")

# --- Fetch weather (API, cached in utils) ---
df_met = fetch_openmeteo(area, year)

tab1, tab2 = st.tabs(["Outlier / SPC (Temp)", "Anomaly / LOF (Precip)"])

with tab1:
    st.subheader("Temperature (°C)")
    st.line_chart(df_met.set_index("time")["temperature_2m"], use_container_width=True)

with tab2:
    st.subheader("Precipitation (mm)")
    st.line_chart(df_met.set_index("time")["precipitation"], use_container_width=True)

with st.expander("Notes"):
    st.markdown("""
- Data source: Open-Meteo ERA5 (hourly). UTC timestamps.
- Area selection shared from page 2 (`st.session_state["price_area"]`).
- This page will later show:
  - **DCT + SPC** (temperature outliers),
  - **LOF** anomalies (precipitation).
    """)
