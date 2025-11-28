# streamlit_app/Home.py
import streamlit as st

st.set_page_config(
    page_title="IND320 – Final Dashboard",
    layout="wide"
)

st.title("IND320 – Data to Decision")
st.subheader("Final Portfolio: Elhub, Open-Meteo, Geospatial Analysis and Forecasting")

st.markdown(
    """
This dashboard is the final outcome of the IND320 course project.  
It combines Elhub energy data, Open-Meteo ERA5 reanalysis, geospatial information for Norwegian price areas (NO1–NO5),  
and several analytical methods: STL decomposition, spectrograms, anomaly detection, snow drift modelling,  
sliding window correlations and SARIMAX forecasting.
"""
)

# --- Global context preview (if already set elsewhere) ---
price_area = st.session_state.get("price_area")
selected_group = st.session_state.get("energy_group")
map_lat = st.session_state.get("map_lat")
map_lon = st.session_state.get("map_lon")

context_lines = []
if price_area:
    context_lines.append(f"- Active price area: **{price_area}**")
if selected_group:
    context_lines.append(f"- Active energy group: **{selected_group}**")
if map_lat is not None and map_lon is not None:
    context_lines.append(f"- Selected map coordinate: **({map_lat:.3f}, {map_lon:.3f})**")

if context_lines:
    st.info("Current shared selections:\n" + "\n".join(context_lines))
else:
    st.info("No global selections yet. Start with the exploration or map pages to define a context.")

st.markdown("---")

st.markdown("### Dashboard structure")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
**1. Exploratory analysis**

- `1_Exploration_Overview`  
  Overview tables and interactive line plots of Elhub data, by price area, group and time period.
- `2_STL_and_Spectrogram`  
  STL decomposition (trend / seasonality / residual) and spectrograms for hourly production data.

**2. Data quality and anomalies**

- `3_Anomalies`  
  Temperature quality control using DCT-based SPC limits and precipitation anomaly detection  
  using Local Outlier Factor (LOF) on Open-Meteo ERA5 data.
"""
    )

with col2:
    st.markdown(
        """
**3. Geospatial view and snow drift**

- `4_Map_and_Choropleth`  
  Map of Elspot price areas (NO1–NO5) based on GeoJSON.  
  Price areas are coloured by mean production or consumption over a user-defined time interval.  
  Map clicks are stored and reused on other pages.
- `5_Snow_Drift`  
  Snow drift per year (July–June definition) for a selected coordinate,  
  plus a wind rose built from ERA5 wind speed and direction.

**4. Correlation and forecasting**

- `6_Correlation_and_Forecast`  
  Sliding window correlations between energy variables and meteorology,  
  and a SARIMAX interface where users can choose training period, forecast horizon  
  and model parameters, including exogenous weather variables.

**5. About**

- `7_About`  
  Project summary, data sources, implementation notes and AI usage.
"""
    )

st.markdown("---")

st.markdown(
    """
### How to use this dashboard

1. Start with **Exploration** to get an intuition for the data.  
2. Use the **Map** page to explore spatial patterns and select a coordinate.  
3. Open the **Snow Drift** page to see yearly snow accumulation and wind patterns at that location.  
4. Explore **Anomalies** to inspect data quality issues in temperature and precipitation.  
5. Finish with **Correlation & Forecast** to investigate relationships and build simple forecasts.

All heavy computations are backed by MongoDB and Open-Meteo ERA5.  
Where possible, results are cached to keep the app responsive during the demo.
"""
)

with st.expander("Links and technical details"):
    st.markdown(
        """
- GitHub repository: `JulesSylMUAMBA/IND320_DataTo_Decision`  
- Data sources:
  - Elhub: production and consumption per group, hourly, price areas NO1–NO5  
  - Open-Meteo ERA5 archive: hourly meteorological variables  
- Back-end:
  - MongoDB Atlas (Elhub data)
  - Cassandra (for large-scale storage, used in the notebooks)
- Front-end:
  - Streamlit with Plotly for all interactive plots.
"""
    )

st.success("Use the left sidebar to navigate between pages. In the presentation, you can follow the structure above as a story line.")
