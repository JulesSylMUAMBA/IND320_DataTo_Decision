import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

@st.cache_data
def load_data():
    # CSV is in the parent folder of "pages/"
    file_path = os.path.join(os.path.dirname(__file__), "..", "open-meteo-subset.csv")
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

st.title("Plots")

# Get list of unique months from the time column
df["month"] = df["time"].dt.to_period("M")
months = sorted(df["month"].unique().astype(str))

col_choice = st.selectbox("Choose a column to plot", ["All"] + list(df.columns[1:5]))  # skip 'time' and 'month'

# Slider to select a range of months
month_range = st.select_slider(
    "Select a range of months",
    options=months,
    value=(months[0], months[0])  # default = first month
)

# Checkbox to enable smoothing
smooth = st.checkbox("Apply smoothing (7-day rolling mean)", value=False)

# Filter data based on selected months
mask = (df["month"] >= month_range[0]) & (df["month"] <= month_range[1])
filtered = df[mask].copy()

# Apply rolling mean if smoothing is enabled
if smooth:
    for col in df.columns[1:5]:
        filtered[col] = filtered[col].rolling(window=7, min_periods=1).mean()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

if col_choice == "All":
    for col in df.columns[1:5]:
        ax.plot(filtered["time"], filtered[col], label=col)
    ax.legend()
else:
    ax.plot(filtered["time"], filtered[col_choice], label=col_choice)
    ax.legend()

ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title(f"Data plot – {col_choice if col_choice != 'All' else 'All variables'}")
plt.xticks(rotation=45)
st.pyplot(fig)
