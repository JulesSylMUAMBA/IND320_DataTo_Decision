# pages/2_STL_and_Spectrogram.py
import streamlit as st
import pandas as pd

from utils import (
    CITY_MAP,
    get_db_and_collection,
    plot_stl_plotly,
    plot_spectrogram_elhub,
)

st.set_page_config(page_title="STL & Spectrogram", layout="wide")
st.title("STL decomposition & Spectrogram")
st.caption("Elhub hourly production data: trend/seasonal/residual decomposition and time–frequency analysis.")

# --- Price area from global context (set on Exploration page); fallback to NO1 ---
area = st.session_state.get("price_area", "NO1")
city = CITY_MAP.get(area, {}).get("city", "Unknown")
st.caption(f"Current area: **{area}** — {city}")

# --- Load Elhub data from MongoDB (cached) ---
@st.cache_data(show_spinner=False, ttl=600)
def load_elhub_df() -> pd.DataFrame:
    db, coll = get_db_and_collection()  # default: production_mba_hour
    docs = list(coll.find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    if not df.empty and "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    return df

df_elhub = load_elhub_df()
if df_elhub.empty:
    st.error("No Elhub data found in MongoDB. Please check the 'production_mba_hour' collection.")
    st.stop()

# --- Available production groups for the selected area ---
groups = (
    df_elhub.loc[df_elhub["price_area"] == area, "production_group"]
    .dropna()
    .unique()
    .tolist()
)
groups = sorted([g for g in groups if g != "*"])

if not groups:
    st.warning(f"No production groups found for area {area}. Try another price area on the Exploration page.")
    st.stop()

default_group = st.session_state.get("energy_group", groups[0])
if default_group not in groups:
    default_group = groups[0]

group = st.selectbox("Production group", groups, index=groups.index(default_group))
st.session_state["energy_group"] = group  # update global context

tab_stl, tab_spec = st.tabs(["STL decomposition", "Spectrogram"])

# -------------------------------------------------------------------
# STL tab
# -------------------------------------------------------------------
with tab_stl:
    st.subheader("STL decomposition (LOESS)")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        period = st.number_input(
            "Seasonal period (samples)",
            min_value=24,
            max_value=24 * 14,
            value=24 * 7,
            step=24,
            help="Typical value: 24×7 = weekly seasonality on hourly data.",
        )
    with colB:
        seasonal = st.number_input(
            "Seasonal smoother (odd ≥ 7)",
            min_value=7,
            max_value=199,
            value=13,
            step=2,
        )
    with colC:
        trend = st.number_input(
            "Trend smoother (odd, > period)",
            min_value=int(period) + 1,
            max_value=999,
            value=max(int(period) + 1, 201),
            step=2,
        )
    with colD:
        robust = st.checkbox("Robust", value=True)

    try:
        fig_stl, res_stl, info_stl = plot_stl_plotly(
            df_elhub,
            price_area=area,
            production_group=group,
            period=int(period),
            seasonal=int(seasonal),
            trend=int(trend),
            robust=robust,
        )
        st.plotly_chart(fig_stl, use_container_width=True)
        with st.expander("Run details"):
            st.json(info_stl)
    except Exception as e:
        st.error(f"STL error: {e}")

# -------------------------------------------------------------------
# Spectrogram tab
# -------------------------------------------------------------------
with tab_spec:
    st.subheader("Spectrogram (time–frequency analysis)")

    col1, col2 = st.columns(2)
    with col1:
        window_len = st.slider(
            "Window length (hours)",
            min_value=48,
            max_value=24 * 30,
            value=24 * 14,
            step=24,
            help="Longer windows give better frequency resolution, shorter windows better time resolution.",
        )
    with col2:
        overlap = st.slider(
            "Overlap",
            min_value=0.0,
            max_value=0.9,
            value=0.5,
            step=0.1,
        )

    try:
        fig_spec, info_spec = plot_spectrogram_elhub(
            df_elhub,
            price_area=area,
            production_group=group,
            window_len=int(window_len),
            overlap=float(overlap),
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        with st.expander("Run details"):
            st.json(info_spec)
    except Exception as e:
        st.error(f"Spectrogram error: {e}")
