import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px

from utils import fetch_price_areas, fetch_groups, fetch_line_data

st.set_page_config(page_title="Exploration overview", layout="wide")

st.title("Exploration overview – Elhub data")

st.markdown(
    """
Use this page to explore Elhub production and consumption data by price area, group and month.  
The data are loaded from MongoDB (hourly resolution).
"""
)

# -----------------------------
# Controls
# -----------------------------
dataset = st.radio(
    "Dataset",
    ["Production", "Consumption"],
    index=0,
    horizontal=True,
)

coll_name = "production_mba_hour" if dataset == "Production" else "consumption_mba_hour"
group_field = "production_group" if dataset == "Production" else "consumption_group"

# Available price areas and groups from MongoDB
areas = fetch_price_areas(coll_name=coll_name)
if not areas:
    st.error("No price areas found in MongoDB. Check the database connection and collections.")
    st.stop()

groups = fetch_groups(coll_name=coll_name, group_field=group_field)
if not groups:
    st.error(f"No {group_field} values found in collection '{coll_name}'.")
    st.stop()

col_sel1, col_sel2, col_sel3 = st.columns([1.2, 1.2, 1])

with col_sel1:
    selected_area = st.selectbox("Price area", areas, index=areas.index("NO1") if "NO1" in areas else 0)

with col_sel2:
    # Default to first few non-star groups if available
    default_groups = [g for g in groups if g != "*"][:3] or groups[:1]
    selected_groups = st.multiselect(
        "Energy groups",
        options=groups,
        default=default_groups,
        help=f"{group_field} values from MongoDB.",
    )

with col_sel3:
    year = st.selectbox("Year", [2023, 2024], index=0)
    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=0,
        format_func=lambda m: datetime(2000, m, 1).strftime("%B"),
    )

# -----------------------------
# Data loading (cached)
# -----------------------------
@st.cache_data(ttl=600)
def load_elhub_month(coll_name: str, area: str, groups: tuple, year: int, month: int, group_field: str):
    if not groups:
        return pd.DataFrame()
    return fetch_line_data(
        price_area=area,
        groups=list(groups),
        year=year,
        month=month,
        coll_name=coll_name,
        group_field=group_field,
    )

df_month = load_elhub_month(
    coll_name=coll_name,
    area=selected_area,
    groups=tuple(selected_groups),
    year=year,
    month=month,
    group_field=group_field,
)

# -----------------------------
# Display
# -----------------------------
if df_month.empty:
    st.warning(
        "No data returned for this combination of area, groups, year and month. "
        "Try a different month or group."
    )
else:
    # Save context in session_state for other pages (Home, forecast, etc.)
    st.session_state["price_area"] = selected_area
    if selected_groups:
        # store a primary group for other pages, but keep the list too
        st.session_state["energy_group"] = selected_groups[0]
        st.session_state["energy_groups"] = selected_groups

    st.subheader("Sample of hourly data")
    st.dataframe(
        df_month.head(50),
        use_container_width=True,
    )

    st.subheader("Time series by group")
    fig = px.line(
        df_month,
        x=df_month.index,
        y=df_month.columns,
        labels={"value": "Quantity (kWh)", "index": "Time (UTC)"},
    )
    fig.update_layout(
        legend_title=group_field,
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
This page provides the basic context for the other parts of the dashboard:  
selected price area and groups are reused in several analysis pages.
"""
)
