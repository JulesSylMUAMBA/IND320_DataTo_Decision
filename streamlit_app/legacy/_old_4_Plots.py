# streamlit_app/pages/4_Plots.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

@st.cache_data
def load_data():
    # CSV in the parent folder of "pages/"
    file_path = os.path.join(os.path.dirname(__file__), "..", "open-meteo-subset.csv")
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

st.title("Interactive Plots (Plotly)")

# Get list of unique months
df["month"] = df["time"].dt.to_period("M")
months = sorted(df["month"].unique().astype(str))

col_choice = st.selectbox("Choose a variable to plot", ["All"] + list(df.columns[1:5]))

# Slider to select range of months
month_range = st.select_slider(
    "Select a range of months",
    options=months,
    value=(months[0], months[-1])
)

# Checkbox to enable smoothing
smooth = st.checkbox("Apply smoothing (7-day rolling mean)", value=False)

# Filter data
mask = (df["month"] >= month_range[0]) & (df["month"] <= month_range[1])
filtered = df.loc[mask].copy()

# Apply rolling mean if smoothing is enabled
if smooth:
    for col in df.columns[1:5]:
        filtered[col] = filtered[col].rolling(window=7, min_periods=1).mean()

# --- Plotly figure ---
fig = go.Figure()

if col_choice == "All":
    for col in df.columns[1:5]:
        fig.add_trace(go.Scatter(
            x=filtered["time"],
            y=filtered[col],
            mode="lines",
            name=col
        ))
else:
    fig.add_trace(go.Scatter(
        x=filtered["time"],
        y=filtered[col_choice],
        mode="lines",
        name=col_choice
    ))

fig.update_layout(
    title=f"Weather Data – {col_choice if col_choice != 'All' else 'All variables'}",
    xaxis_title="Time",
    yaxis_title="Value",
    hovermode="x unified",
    template="plotly_white",
    height=500,
    margin=dict(l=10, r=10, t=60, b=10)
)

st.plotly_chart(fig, use_container_width=True)
