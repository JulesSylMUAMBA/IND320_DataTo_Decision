# pages/6_Outliers_Anomalies.py
import streamlit as st
from utils import (
    fetch_openmeteo,
    CITY_MAP,
    plot_temp_spc_plotly,
    plot_precip_lof_plotly,
)

st.set_page_config(page_title="Outliers & Anomalies", layout="wide")
st.title("New B — Outlier/SPC & Anomaly/LOF")

# --- Area from global selector (page 2); fallback local ---
area = st.session_state.get("price_area")
if area is None:
    area = st.selectbox("Price area (local)", list(CITY_MAP.keys()), index=4)  # NO5 by default
city = CITY_MAP[area]["city"]

# Year fixed to 2021 as required
year = 2021
st.caption(f"Downloading ERA5 for **{city} ({area})**, year **{year}** (fixed).")

# --- Fetch weather (API, cached in utils) ---
df_met = fetch_openmeteo(area, year)

tab1, tab2 = st.tabs(["Outlier / SPC (Temp)", "Anomaly / LOF (Precip)"])

# ----------------------------------------------------------------------
# TAB 1 : Temperature — DCT + SPC
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Temperature — DCT + SPC")

    col1, col2, col3 = st.columns(3)
    with col1:
        cutoff = st.slider(
            "DCT cutoff (fraction of lowest frequencies removed)",
            0.0, 0.2, 0.05, 0.01
        )
    with col2:
        n_sigma = st.slider(
            "SPC sigma (std multiples)",
            2.0, 5.0, 3.0, 0.5
        )
    with col3:
        robust = st.checkbox("Robust (median/MAD)", value=True)

    fig_spc, summary_spc = plot_temp_spc_plotly(
        df_met,
        cutoff=cutoff,
        n_sigma=n_sigma,
        robust=robust,
    )
    st.plotly_chart(fig_spc, use_container_width=True)
    st.markdown("**Outlier summary:**")
    st.json(summary_spc)

# ----------------------------------------------------------------------
# TAB 2 : Precipitation — LOF anomalies
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Precipitation — LOF anomalies")

    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider(
            "Contamination (outlier proportion)",
            0.001, 0.05, 0.01, 0.001
        )
    with col2:
        n_neighbors = st.slider(
            "n_neighbors (LOF)",
            10, 60, 40, 1
        )

    fig_lof, summary_lof, anomalies = plot_precip_lof_plotly(
        df_met,
        contamination=float(contamination),
        n_neighbors=int(n_neighbors),
    )
    st.plotly_chart(fig_lof, use_container_width=True)
    st.markdown("**Anomaly summary:**")
    st.json(summary_lof)

    with st.expander("Anomalies (top 20)"):
        st.dataframe(anomalies.head(20))

with st.expander("Notes"):
    st.markdown("""
- **Source:** Open-Meteo ERA5 (hourly, UTC).
- **SPC tab:** DCT high-pass → seasonally adjusted temperature variations (SATV) → SPC limits → outliers highlighted.
- **LOF tab:** Local Outlier Factor on precipitation with robust scaling and tunable contamination.
    """)