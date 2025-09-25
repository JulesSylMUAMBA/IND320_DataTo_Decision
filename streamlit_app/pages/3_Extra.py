import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("open-meteo-subset.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

st.title("Extra Analysis – Correlation Matrix")

st.write("""
This page shows the correlation matrix between the different variables
(temperature, precipitation, wind speed, wind gusts, and wind direction).
Correlations range from -1 (perfect negative) to +1 (perfect positive).
""")

# Compute correlation
corr = df.drop(columns=["time"]).corr()

# Plot heatmap with matplotlib
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax)

# Set ticks and labels
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="left")
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)

plt.title("Correlation Matrix", pad=20)
st.pyplot(fig)

# Show numeric values
st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None))
