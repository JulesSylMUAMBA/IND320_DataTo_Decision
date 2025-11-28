# pages/3_Anomalies.py
import streamlit as st
from utils import (
    fetch_openmeteo,
    CITY_MAP,
    plot_temp_spc_plotly,
    plot_precip_lof_plotly,
)

st.set_page_config(page_title="Anomalies & SPC", layout="wide")
st.title("Anomalies & Quality Control (SPC / LOF)")

st.markdown(
    """
This page focuses on **data quality and anomalies** in the meteorological data (Open-Meteo ERA5).

- Temperature: DCT-based SPC limits with outliers highlighted.  
- Precipitation: anomaly detection using Local Outlier Factor (LOF).
"""
)

# --------------------------------------------------------------------
# Global context: price area + year
# --------------------------------------------------------------------
col_area, col_year = st.columns(2)

with col_area:
    # Préférence : utiliser la zone déjà choisie ailleurs, mais laisser l’utilisateur changer
    default_area = st.session_state.get("price_area", "NO1")
    areas = list(CITY_MAP.keys())
    if default_area not in areas:
        default_area = "NO1"

    area = st.selectbox(
        "Price area",
        areas,
        index=areas.index(default_area),
    )
    st.session_state["price_area"] = area  # mise à jour du contexte global
    city = CITY_MAP[area]["city"]

with col_year:
    # On laisse un peu de liberté sur l’année (utile pour l’oral)
    year = st.selectbox(
        "Year (ERA5 archive)",
        [2021, 2022, 2023, 2024],
        index=0,
    )

st.caption(f"Using ERA5 data for **{city} ({area})**, year **{year}**.")

# --------------------------------------------------------------------
# Fetch weather data (cached in utils)
# --------------------------------------------------------------------
df_met = fetch_openmeteo(area, year)
if df_met is None or df_met.empty:
    st.error("No ERA5 data returned for this selection. Check the API or try another year/area.")
    st.stop()

tab_spc, tab_lof = st.tabs(["Temperature SPC (DCT)", "Precipitation LOF"])

# ----------------------------------------------------------------------
# TAB 1 : Temperature — DCT + SPC
# ----------------------------------------------------------------------
with tab_spc:
    st.subheader("Temperature — DCT-based SPC")

    col1, col2, col3 = st.columns(3)
    with col1:
        cutoff = st.slider(
            "DCT cutoff (fraction of lowest frequencies removed)",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Controls how much low-frequency (seasonal) variation is removed.",
        )
    with col2:
        n_sigma = st.slider(
            "SPC sigma (standard deviations)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Outliers are points where the adjusted signal leaves these bands.",
        )
    with col3:
        robust = st.checkbox(
            "Robust (median/MAD)",
            value=True,
            help="If checked, SPC limits are based on median/MAD instead of mean/std.",
        )

    try:
        fig_spc, summary_spc = plot_temp_spc_plotly(
            df_met,
            time_col="time",
            temp_col="temperature_2m",
            cutoff=cutoff,
            n_sigma=n_sigma,
            robust=robust,
        )
        st.plotly_chart(fig_spc, use_container_width=True)

        st.markdown("**Outlier summary:**")
        st.json(summary_spc)
    except Exception as e:
        st.error(f"SPC computation error: {e}")

# ----------------------------------------------------------------------
# TAB 2 : Precipitation — LOF anomalies
# ----------------------------------------------------------------------
with tab_lof:
    st.subheader("Precipitation — LOF anomalies")

    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider(
            "Contamination (expected outlier proportion)",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
        )
    with col2:
        n_neighbors = st.slider(
            "n_neighbors (LOF)",
            min_value=10,
            max_value=60,
            value=40,
            step=1,
        )

    try:
        fig_lof, summary_lof, anomalies = plot_precip_lof_plotly(
            df_met,
            time_col="time",
            precip_col="precipitation",
            contamination=float(contamination),
            n_neighbors=int(n_neighbors),
        )
        st.plotly_chart(fig_lof, use_container_width=True)

        st.markdown("**Anomaly summary:**")
        st.json(summary_lof)

        with st.expander("Anomalies (top 20 by LOF score)"):
            st.dataframe(anomalies.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"LOF computation error: {e}")

# ----------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------
with st.expander("Notes"):
    st.markdown(
        """
- **Source:** Open-Meteo ERA5 archive (hourly data, UTC).
- **Temperature SPC:**  
  - A DCT (Discrete Cosine Transform) high-pass filter is used to remove slow seasonal trends.  
  - SPC limits are computed on the seasonally adjusted signal (SATV), then projected back on the raw series.  
  - This ensures that **SPC boundaries follow the overall pattern of the data**, not horizontal lines.
- **Precipitation LOF:**  
  - A robust scaling is applied and a Local Outlier Factor model is fitted.  
  - Points with the highest LOF scores are marked as anomalies.
"""
    )
