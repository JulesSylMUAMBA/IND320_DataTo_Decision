# pages/5_Snow_Drift.py
import streamlit as st
import plotly.express as px

from utils import (
    calculate_snow_drift_range,
    calculate_snow_drift_wind_rose,
    plot_wind_rose,
)

st.set_page_config(page_title="Snow Drift & Wind Rose", page_icon="❄️", layout="wide")

st.title("Snow Drift & Wind Rose")
st.markdown(
    """
This page computes **yearly snow drift** and visualizes the **wind rose**  
using **ERA5 meteorological data** at the location selected on the **Map** page.
"""
)

# --------------------------------------------------------------------
# 1. Retrieve coordinates stored by the Map page
# --------------------------------------------------------------------
lat = st.session_state.get("map_lat")
lon = st.session_state.get("map_lon")

if lat is None or lon is None:
    st.warning(
        "No coordinates selected yet. Please go to the **Map & Price Areas** page, "
        "click on a point, and then come back here."
    )
    st.stop()

st.info(f"Using selected coordinate: **lat={lat:.3f} · lon={lon:.3f}**")

# --------------------------------------------------------------------
# 2. Choose year range + display options
# --------------------------------------------------------------------
col_years, col_opts = st.columns([1.2, 1])

with col_years:
    start_year, end_year = st.select_slider(
        "Year range (hydrological years from 1 July → 30 June)",
        options=[2021, 2022, 2023, 2024],
        value=(2021, 2024),
    )

with col_opts:
    show_monthly = st.checkbox(
        "Show yearly + monthly snow drift (if available)",
        value=False,
        help="If monthly results are computed by calculate_snow_drift_range.",
    )

st.caption(
    f"Hydrological years from **{start_year}–{start_year+1}** up to "
    f"**{end_year}–{end_year+1}**."
)

# --------------------------------------------------------------------
# 3. Compute snow drift over the selected year range
# --------------------------------------------------------------------
st.subheader("Yearly snow drift")

with st.spinner("Computing snow drift for selected years..."):
    try:
        df_years = calculate_snow_drift_range(
            lat=lat,
            lon=lon,
            start_year=start_year,
            end_year=end_year,
        )
    except Exception as e:
        st.error(f"Error while computing snow drift: {e}")
        st.stop()

if df_years is None or df_years.empty:
    st.warning("No snow drift data could be computed for this period and location.")
else:
    # Try to automatically identify the "year" and "value" columns
    cols = list(df_years.columns)

    # 1) year column
    if "year" in cols:
        year_col = "year"
    else:
        # Take the first column that looks like a year
        year_col = cols[0]

    # 2) value column (snow drift)
    if "snow_drift_mm" in cols:
        value_col = "snow_drift_mm"
    else:
        # Take another numeric column as value
        numeric_cols = df_years.select_dtypes("number").columns.tolist()
        # Avoid using the year column as value
        numeric_cols = [c for c in numeric_cols if c != year_col]
        if not numeric_cols:
            st.error("Could not find a numeric snow drift column in the returned DataFrame.")
        else:
            value_col = numeric_cols[0]

    fig_years = px.bar(
        df_years,
        x=year_col,
        y=value_col,
        labels={year_col: "Hydrological year", value_col: "Snow drift (mm)"},
        title="Snow drift per hydrological year",
    )
    fig_years.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_years, use_container_width=True)

    st.markdown("**Yearly summary:**")
    st.dataframe(df_years, use_container_width=True)


# Optional: monthly breakdown if your function returns detailed data
if show_monthly and df_years is not None and "monthly" in df_years.columns:
    st.subheader("Monthly snow drift (experimental)")
    # Example: df_years['monthly'] could be a list of DataFrames or similar.
    st.write(
        "Monthly breakdown depends on how `calculate_snow_drift_range` returns details. "
        "Adjust this block if you expose a monthly DataFrame there."
    )

# --------------------------------------------------------------------
# 4. Wind rose for a specific year
# --------------------------------------------------------------------
st.subheader("Wind rose for a selected year")

wind_year = st.selectbox(
    "Select year for wind rose",
    options=list(range(start_year, end_year + 1)),
    index=0,
)

with st.spinner(f"Computing wind rose for year {wind_year}..."):
    try:
        df_snow_year, wind_df = calculate_snow_drift_wind_rose(
            lat=lat,
            lon=lon,
            year=wind_year,
        )
    except Exception as e:
        st.error(f"Error while computing wind rose data: {e}")
        wind_df = None

if wind_df is None or wind_df.empty:
    st.warning("No wind data available to build the wind rose for this year.")
else:
    try:
        fig_wind = plot_wind_rose(wind_df)
        st.plotly_chart(fig_wind, use_container_width=True)
    except Exception as e:
        st.error(f"Error while plotting wind rose: {e}")

with st.expander("Notes"):
    st.markdown(
        """
- **Coordinate source:** user click on the *Map & Price Areas* page  
  (stored in `st.session_state['map_lat']` and `['map_lon']`).  
- **Year definition:** from **1 July** of the selected year to **30 June** of the next year.  
- **Snow drift model:** simplified accumulation/melt based on ERA5 precipitation and temperature.  
- **Wind rose:** uses wind speed / wind direction from ERA5 at the same coordinate.
"""
    )
