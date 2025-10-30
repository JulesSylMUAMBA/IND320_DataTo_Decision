# pages/4_Production_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import fetch_price_areas, fetch_groups, fetch_pie_data, fetch_line_data

st.set_page_config(page_title="Production Analysis", layout="wide")
st.title("Production Analysis (Elhub)")

st.caption(
    "Data source: Elhub hourly production per group. Stored in MongoDB (UTC)."
)

left, right = st.columns(2)

with left:
    st.subheader("Total production by group (Pie)")
    areas = fetch_price_areas()
    if not areas:
        st.warning("No price areas found in MongoDB.")
    else:
        price_area = st.radio("Choose price area:", areas, index=0, horizontal=True)

        @st.cache_data(show_spinner=False, ttl=300)
        def cached_pie(pa: str) -> pd.DataFrame:
            return fetch_pie_data(pa)

        df_pie = cached_pie(price_area)
        if df_pie.empty:
            st.info("No data for the selected price area.")
        else:
            fig_pie = px.pie(
                df_pie,
                values="quantity_kwh",
                names="production_group",
                title=f"Total production by group – {price_area}",
                hole=0.3,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

with right:
    st.subheader("Hourly production by group (Line)")
    groups = [g for g in fetch_groups() if g != "*"]
    if not groups:
        st.warning("No production groups found in MongoDB.")
    else:
        # --- Use pills if available (Streamlit >= 1.33); otherwise fall back to multiselect ---
        use_pills = hasattr(st, "pills")
        if use_pills:
            selected_groups = st.pills("Select groups:", groups, selection=groups[:3])
        else:
            selected_groups = st.multiselect("Select groups:", groups, default=groups[:3])

        year = st.selectbox("Year:", [2021, 2022, 2023, 2024, 2025], index=0)  # default to 2021 now
        month = st.selectbox("Month:", list(range(1, 13)), index=0, format_func=lambda m: f"{m:02d}")

        if selected_groups:
            @st.cache_data(show_spinner=False, ttl=300)
            def cached_line(pa: str, gs: tuple, yy: int, mm: int) -> pd.DataFrame:
                return fetch_line_data(pa, list(gs), yy, mm)

            df_line = cached_line(price_area, tuple(selected_groups), year, month)
            if df_line.empty:
                st.info("No data found for the selected filters.")
            else:
                fig_line = go.Figure()
                for col in df_line.columns:
                    fig_line.add_trace(go.Scatter(x=df_line.index, y=df_line[col], mode="lines", name=col))
                fig_line.update_layout(
                    title=f"Hourly production – {price_area} – {year}-{month:02d} (UTC)",
                    xaxis_title="Time",
                    yaxis_title="Quantity (kWh)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=480,
                    margin=dict(l=10, r=10, t=60, b=10),
                )
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Select at least one production group.")

with st.expander("Data & methodology"):
    st.markdown("""
- **Source:** Elhub (hourly production per group).
- **Time:** stored as **UTC** in MongoDB. For local display, convert to Europe/Oslo if needed.
- **Fields:** `price_area`, `production_group`, `start_time` (UTC), `quantity_kwh`.
- **Flow:** Notebook → clean → insert into MongoDB → read here.
    """)
