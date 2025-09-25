import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("../data/open-meteo-subset.csv")

df = load_data()

st.title("Data Table")
st.write("Each row corresponds to one variable, with a mini time series (first month).")

# Keep only the first month of data
df["time"] = pd.to_datetime(df["time"])
first_month = df["time"].dt.to_period("M")[0]  # first month period
subset = df[df["time"].dt.to_period("M") == first_month]

# Define units for each variable
units = {
    "temperature_2m (°C)": "°C",
    "precipitation (mm)": "mm",
    "wind_speed_10m (m/s)": "m/s",
    "wind_gusts_10m (m/s)": "m/s",
    "wind_direction_10m (°)": "°"
}

# Reshape: one row per variable (except time)
table = pd.DataFrame({
    "Variable": df.columns[1:],  # skip "time"
    "Unit": [units.get(col, "") for col in df.columns[1:]],
    "Mini-plot": [subset[col].tolist() for col in df.columns[1:]]
})

# Display with LineChartColumn
st.dataframe(
    table,
    column_config={
        "Variable": st.column_config.TextColumn("Variable"),
        "Unit": st.column_config.TextColumn("Unit"),
        "Mini-plot": st.column_config.LineChartColumn("First month")
    },
    hide_index=True,
)
