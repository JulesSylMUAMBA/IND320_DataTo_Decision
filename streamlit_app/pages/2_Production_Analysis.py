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




# --- Split layout: more space on the left for the pie chart ---
left, right = st.columns([1.3, 0.9])  # wider left column

with left:
    # --- Radio selector for price area (as required by assignment) ---
    st.subheader("Total production by group (Pie)")
    st.caption("Select a price area to visualize total annual production.")
    areas = list(CITY_MAP.keys())
    default_area = st.session_state.get("price_area", "NO5")
    area = st.radio(
        "Choose price area:",
        areas,
        index=areas.index(default_area),
        horizontal=True
    )
    st.session_state["price_area"] = area
    st.caption(f"Current area: **{area}** – {CITY_MAP[area]['city']}")

    # --- Fetch and plot pie chart ---
    available_areas = fetch_price_areas()
    if not available_areas:
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

           
            fig_pie.update_traces(
                textinfo="percent+label",
                textposition="inside",
                hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent})<extra></extra>",
                name=""  
            )
            fig_pie.update_layout(
                title_text="",             
                showlegend=True,          
                margin=dict(t=80, b=10, l=20, r=10),
                height=350,
            )

            
            for tr in fig_pie.data:
                tr.name = ""  # ou "Production"

            st.plotly_chart(fig_pie, use_container_width=True)


with right:
    st.subheader("Hourly production by group (Line)")

    # 1) Fetch available production groups (excluding wildcard)
    groups = [g for g in fetch_groups() if g != "*"]
    if not groups:
        st.warning("No production groups found in MongoDB.")
    else:
        # 2) Required by assignment: use st.pills (no fallback)
        if not hasattr(st, "pills"):
            st.error("This app requires Streamlit ≥ 1.38 for st.pills (as per assignment).")
        else:
            # Multi-selection of production groups via pills
            selected_groups = st.pills(
                "Select groups:",
                options=groups,
                default=groups[:3],
                selection_mode="multi"
            )

            # Ensure it's always a list
            if isinstance(selected_groups, str):
                selected_groups = [selected_groups]

            # 3) Month selector (year fixed as per assignment feedback)
            month = st.selectbox(
                "Month:",
                list(range(1, 13)),
                index=0,
                format_func=lambda m: f"{m:02d}"
            )
            year = 2021  # fixed year

            # 4) Show current filters clearly (combine price area + groups + month)
            st.caption(
                f"Area: **{area}** – {CITY_MAP[area]['city']} • "
                f"Month: **{year}-{month:02d}** • "
                f"Groups: {', '.join(selected_groups) if selected_groups else '—'}"
            )

            # 5) Fetch and plot line chart "like in the Jupyter notebook"
            if selected_groups:
                @st.cache_data(show_spinner=False, ttl=300)
                def cached_line(pa: str, gs: tuple, yy: int, mm: int) -> pd.DataFrame:
                    return fetch_line_data(pa, list(gs), yy, mm)

                df_line = cached_line(area, tuple(selected_groups), year, month)

                if df_line.empty:
                    st.info("No data found for the selected filters.")
                else:
                    # Ensure index is datetime (for better x-axis formatting)
                    if not pd.api.types.is_datetime64_any_dtype(df_line.index):
                        try:
                            df_line.index = pd.to_datetime(df_line.index, utc=True)
                        except Exception:
                            pass

                    # Create one line per production group
                    fig_line = go.Figure()
                    for col in df_line.columns:
                        fig_line.add_trace(
                            go.Scatter(
                                x=df_line.index,
                                y=df_line[col],
                                mode="lines",
                                name=col
                            )
                        )

                    # Improve layout and match notebook style
                    fig_line.update_layout(
                        title=f"Hourly prod of {CITY_MAP[area]['city']} in {year}-{month:02d} [UTC]",
                        xaxis_title="Time (UTC)",
                        yaxis_title="Quantity (kWh)",
                        hovermode="x unified",
                        template="plotly_white",
                        height=480,
                        margin=dict(l=10, r=10, t=60, b=10),
                    )

                    # Display chart
                    st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Select at least one production group.")

# --- Data explanation ---
with st.expander("Data & methodology"):
    st.markdown("""
- **Source:** Elhub (hourly production per group).
- **Time:** stored as **UTC** in MongoDB. For local display, convert to Europe/Oslo if needed.
- **Fields:** `price_area`, `production_group`, `start_time` (UTC), `quantity_kwh`.
- **Flow:** Notebook → clean → insert into MongoDB → read here.
    """)
