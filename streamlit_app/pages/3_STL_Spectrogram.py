# pages/3_STL_Spectrogram.py
import streamlit as st
import pandas as pd

from utils import (
    CITY_MAP,
    get_db_and_collection,   # helpers Mongo
)

# Si tes fonctions sont bien dans utils.py, on les importe.
# Sinon, on affichera un message d’erreur propre.
try:
    from utils import plot_stl_plotly, plot_spectrogram_elhub
    _HAS_FUNCS = True
except Exception:
    _HAS_FUNCS = False

st.set_page_config(page_title="STL & Spectrogram", layout="wide")
st.title("New A — STL decomposition & Spectrogram")
st.caption("Elhub production: STL decomposition (trend/seasonal/residual) and time–frequency spectrogram.")

# --- Area from global selector (page 2); fallback to NO5 if absent ---
area = st.session_state.get("price_area", "NO5")
st.caption(f"Current area: **{area}** — {CITY_MAP[area]['city']}")

# --- Load Elhub from MongoDB (simple, cached) ---
@st.cache_data(show_spinner=False, ttl=300)
def load_elhub_df() -> pd.DataFrame:
    db, coll = get_db_and_collection()
    docs = list(coll.find({}, {"_id": 0}))
    return pd.DataFrame(docs)

df_elhub = load_elhub_df()
if df_elhub.empty:
    st.warning("No Elhub data found in MongoDB collection. Please load data first.")
    st.stop()

# Ensure types
if "start_time" in df_elhub.columns:
    df_elhub["start_time"] = pd.to_datetime(df_elhub["start_time"], utc=True, errors="coerce")

# --- Available groups for the selected area ---
groups = sorted(
    df_elhub.loc[df_elhub["price_area"] == area, "production_group"]
    .dropna()
    .unique()
    .tolist()
)
if not groups:
    st.warning(f"No production groups found for area {area}.")
    st.stop()

group = st.selectbox("Production group", groups, index=0)

tab1, tab2 = st.tabs(["STL", "Spectrogram"])

with tab1:
    st.subheader("STL decomposition (LOESS)")

    # Controls
    colA, colB, colC, colD = st.columns(4)
    with colA:
        period = st.number_input("Seasonal period (samples)", min_value=24, max_value=24*14, value=24*7, step=24)
    with colB:
        seasonal = st.number_input("Seasonal smoother (odd ≥ 7)", min_value=7, max_value=199, value=13, step=2)
    with colC:
        trend = st.number_input("Trend smoother (odd, > period)", min_value=int(period)+1, max_value=999, value=max(int(period)+1, 201), step=2)
    with colD:
        robust = st.checkbox("Robust", value=True)

    if not _HAS_FUNCS:
        st.error("plot_stl_plotly is not available. Please move the function into utils.py.")
    else:
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
            with st.expander("Run info"):
                st.json(info_stl)
        except Exception as e:
            st.error(f"STL error: {e}")

with tab2:
    st.subheader("Spectrogram (time–frequency)")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        window_len = st.slider("Window length (hours)", 48, 24*30, 24*14, 24)
    with col2:
        overlap = st.slider("Overlap", 0.0, 0.9, 0.5, 0.1)

    if not _HAS_FUNCS:
        st.error("plot_spectrogram_elhub is not available. Please move the function into utils.py.")
    else:
        try:
            fig_spec, info_spec = plot_spectrogram_elhub(
                df_elhub,
                price_area=area,
                production_group=group,
                window_len=int(window_len),
                overlap=float(overlap),
            )
            st.plotly_chart(fig_spec, use_container_width=True)
            with st.expander("Run info"):
                st.json(info_spec)
        except Exception as e:
            st.error(f"Spectrogram error: {e}")
