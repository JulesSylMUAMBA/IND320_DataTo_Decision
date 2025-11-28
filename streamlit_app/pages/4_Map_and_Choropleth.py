import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta

import folium
from utils import (
    load_elspot_geojson,
    fetch_mean_by_price_area,
    fetch_price_areas,
    fetch_groups,
    CITY_MAP,
)

# Optional: richer interactivity if streamlit-folium is installed
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

st.set_page_config(page_title="Map & Price Areas", page_icon="🗺️", layout="wide")

st.title("Map & Price Areas (Choropleth)")
st.markdown(
    """
This page displays the Norwegian Elspot price areas (NO1–NO5) and colors them according 
to **mean production or consumption** over a selectable time interval.

Clicking on the map stores the coordinate in the session so it can be reused on the **Snow Drift** page.
"""
)

# --------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------
col_dataset, col_group, col_period = st.columns([1.1, 1.3, 1.6])

with col_dataset:
    dataset_type = st.radio(
        "Dataset",
        ["Production", "Consumption"],
        horizontal=True,
    )
    if dataset_type == "Production":
        collection_name = "production_mba_hour"
        group_field = "production_group"
    else:
        collection_name = "consumption_mba_hour"
        group_field = "consumption_group"

with col_group:
    # Group options from MongoDB
    groups = fetch_groups(coll_name=collection_name, group_field=group_field)
    groups = [g for g in groups if g != "*"]
    if not groups:
        st.error(f"No {group_field} values found in MongoDB collection '{collection_name}'.")
        st.stop()

    default_group = st.session_state.get("energy_group", groups[0])
    if default_group not in groups:
        default_group = groups[0]

    group = st.selectbox(
        f"{group_field.replace('_', ' ').capitalize()}",
        groups,
        index=groups.index(default_group),
    )
    st.session_state["energy_group"] = group

with col_period:
    # Start date + number of days (time interval in days)
    start_date = st.date_input(
        "Start date (UTC)",
        value=datetime(2023, 1, 1).date(),
        min_value=datetime(2023, 1, 1).date(),
        max_value=datetime(2024, 12, 31).date(),
    )
    n_days = st.slider(
        "Interval length (days)",
        min_value=1,
        max_value=60,
        value=7,
    )

start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
end_dt = start_dt + timedelta(days=n_days)

st.caption(
    f"Interval: **{start_dt.date()}** → **{(end_dt - timedelta(seconds=1)).date()}** "
    f"({n_days} day(s)), dataset: **{dataset_type}**, group: **{group}**."
)

# Highlighted area (for border emphasis)
areas_available = fetch_price_areas(coll_name=collection_name)
if not areas_available:
    st.error("No price areas found in MongoDB.")
    st.stop()

default_area = st.session_state.get("price_area", "NO1")
if default_area not in areas_available:
    default_area = areas_available[0]

highlight_area = st.selectbox(
    "Highlight price area (border emphasis)",
    areas_available,
    index=areas_available.index(default_area),
)
st.session_state["price_area"] = highlight_area

# --------------------------------------------------------------------
# Fetch mean values per price area
# --------------------------------------------------------------------
stats_df = fetch_mean_by_price_area(
    coll_name=collection_name,
    group_field=group_field,
    group=group,
    start_time_utc=start_dt,
    end_time_utc=end_dt,
)

if stats_df.empty:
    st.warning("No data found for this combination of dataset, group and period.")
    st.stop()

# Adapt area codes: "NO1" -> "NO 1" to match the GeoJSON ElSpotOmr field
stats_df["elspot_code"] = stats_df["price_area"].apply(
    lambda pa: f"{pa[:2]} {pa[2:]}" if isinstance(pa, str) and len(pa) == 3 else pa
)

# --------------------------------------------------------------------
# Load GeoJSON
# --------------------------------------------------------------------
geojson = load_elspot_geojson()
if not geojson:
    st.error("GeoJSON for Elspot areas could not be loaded. Check the file path in utils.load_elspot_geojson().")
    st.stop()

# st.write("Example GeoJSON properties:", geojson["features"][0]["properties"])

# --------------------------------------------------------------------
# Folium map
# --------------------------------------------------------------------
# Center somewhere over Norway using the CITY_MAP coordinates
center_lat = sum(v["lat"] for v in CITY_MAP.values()) / len(CITY_MAP)
center_lon = sum(v["lon"] for v in CITY_MAP.values()) / len(CITY_MAP)

m = folium.Map(location=[center_lat, center_lon], zoom_start=4.7, tiles="cartodbpositron")

# Choropleth: color by mean_quantity_kwh
# Assumption: each feature has a property 'ElSpotOmr' like "NO 1", "NO 2", etc.
choropleth = folium.Choropleth(
    geo_data=geojson,
    name="choropleth",
    data=stats_df,
    columns=["elspot_code", "mean_quantity_kwh"],
    key_on="feature.properties.ElSpotOmr",
    fill_color="YlGnBu",
    fill_opacity=0.6,
    line_opacity=0.8,
    highlight=True,
    legend_name=f"Mean {dataset_type} (kWh) over interval",
).add_to(m)

# Add labels / tooltips using the ElSpotOmr property
folium.GeoJsonTooltip(
    fields=["ElSpotOmr"],
    aliases=["Price area:"],
).add_to(choropleth.geojson)

# Extra overlay with thicker border for the highlighted area
def style_function(feature):
    # ElSpotOmr is like "NO 2", "NO 3", etc.
    pa_raw = feature["properties"].get("ElSpotOmr")
    # Convert "NO 2" -> "NO2" to compare with highlight_area ("NO1", "NO2", etc.)
    pa = pa_raw.replace(" ", "") if isinstance(pa_raw, str) else pa_raw

    if pa == highlight_area:
        return {
            "fillOpacity": 0.0,
            "color": "red",
            "weight": 3,
        }
    else:
        return {
            "fillOpacity": 0.0,
            "color": "black",
            "weight": 1,
        }

folium.GeoJson(
    geojson,
    name="highlight",
    style_function=style_function,
).add_to(m)

# If we have previous click coordinates, add a marker
click_lat = st.session_state.get("map_lat")
click_lon = st.session_state.get("map_lon")
if click_lat is not None and click_lon is not None:
    folium.Marker(
        location=[click_lat, click_lon],
        popup=f"Selected point: ({click_lat:.3f}, {click_lon:.3f})",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

folium.LayerControl().add_to(m)

st.markdown("### Interactive map")

if HAS_ST_FOLIUM:
    map_data = st_folium(m, width=900, height=600)

    # Store last clicked coordinates (if any) in session_state
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.session_state["map_lat"] = lat
        st.session_state["map_lon"] = lon

        st.success(
            f"Stored clicked coordinate: ({lat:.3f}, {lon:.3f}). "
            f"You can now use it on the Snow Drift page."
        )
else:
    st.warning(
        "Package `streamlit-folium` is not installed, so click events cannot be captured.\n\n"
        "Install it and add it to `requirements.txt` to enable fully interactive behaviour."
    )
    # Fallback: render static HTML
    from streamlit.components.v1 import html
    html(m._repr_html_(), height=600)

# --------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------
with st.expander("Summary of mean values per price area"):
    st.dataframe(
        stats_df.rename(columns={"mean_quantity": "mean_kWh", "n_samples": "n_hours"}),
        use_container_width=True,
    )

st.markdown(
    """
The selected coordinate (if any) is stored in `st.session_state['map_lat']` and 
`st.session_state['map_lon']` and reused in the **Snow Drift** page for local analysis.
"""
)
