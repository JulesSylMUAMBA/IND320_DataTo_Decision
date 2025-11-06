# pages/2_Production_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    fetch_price_areas,
    fetch_groups,
    fetch_pie_data,
    fetch_line_data,
    CITY_MAP,
)

st.title("Production Analysis (Elhub)")
st.caption("Data source: Elhub hourly production per group. Stored in MongoDB (UTC).")

# --- Global area selector (shared with other pages) ---
default_area = st.session_state.get("price_area", "NO5")
area = st.sidebar.selectbox(
    "Price area",
    list(CITY_MAP.keys()),
    index=list(CITY_MAP.keys()).index(default_area),
)
st.session_state["price_area"] = area
st.caption(f"Current area: **{area}** – {CITY_MAP[area]['city']}")

left, right = st.columns(2)

with left:
    st.subheader("Total production by group (Pie)")
    areas = fetch_price_areas()
    if not areas:
        st.warning("No price areas found in MongoDB.")
    else:
        @st.cache_data(show_spinner=False, ttl=300)
        def cached_pie(pa: str) -> pd.DataFrame:
            return fetch_pie_data(pa)

        df_pie = cached_pie(area)
        if df_pie.empty:
            st.info("No data for the selected price area.")
        else:
            fig_pie = px.pie(
                df_pie,
                values="quantity_kwh",
                names="production_group",
                hole=0.3,
            )
            fig_pie.update_layout(
                title=None,
                margin=dict(t=100, b=50, l=10, r=10),
                height=500,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)


with right:
    st.subheader("Hourly production by group (Line)")
    groups = [g for g in fetch_groups() if g != "*"]
    if not groups:
        st.warning("No production groups found in MongoDB.")
    else:
        # --- Group selection UI (robust across Streamlit versions) ---
        selected_groups = None

        if hasattr(st, "pills"):
            # Try with multi-selection enabled (some versions require it for list defaults)
            try:
                selected_groups = st.pills(
                    "Select groups:", groups,
                    default=groups[:3],
                    selection_mode="multi"
                )
            except TypeError:
                # Older signature without selection_mode
                try:
                    selected_groups = st.pills("Select groups:", groups, default=groups[:3])
                except Exception:
                    selected_groups = None
            except Exception:
                selected_groups = None

        # Fallback to multiselect if pills is unavailable or failed
        if selected_groups is None:
            selected_groups = st.multiselect("Select groups:", groups, default=groups[:3])

        # Normalize to list if a single string is returned
        if isinstance(selected_groups, str):
            selected_groups = [selected_groups]


        month = st.selectbox("Month:", list(range(1, 13)), index=0, format_func=lambda m: f"{m:02d}")
        year = 2021  # ask by review


        if selected_groups:
            @st.cache_data(show_spinner=False, ttl=300)
            def cached_line(pa: str, gs: tuple, yy: int, mm: int) -> pd.DataFrame:
                return fetch_line_data(pa, list(gs), yy, mm)

            df_line = cached_line(area, tuple(selected_groups), year, month)
            if df_line.empty:
                st.info("No data found for the selected filters.")
            else:
                fig_line = go.Figure()
                for col in df_line.columns:
                    fig_line.add_trace(
                        go.Scatter(x=df_line.index, y=df_line[col], mode="lines", name=col)
                    )
                fig_line.update_layout(
                    title=f"Hourly production – {area} – {year}-{month:02d} (UTC)",
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
