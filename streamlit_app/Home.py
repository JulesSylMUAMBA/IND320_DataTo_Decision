import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="IND320 Project", layout="wide")

# Function to load data safely from the same folder as Home.py
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "open-meteo-subset.csv")
    return pd.read_csv(file_path)

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the sidebar to navigate between pages.")

# Main content
st.title("IND320 – Data to Decision")
st.write("""
Welcome to my Streamlit project for the IND320 course.  
My name is **Jules**, and I am excited to share my work with you.  

This app allows you to explore weather data interactively:
- **Page 1 (Table):** Dataset with mini time series  
- **Page 2 (Plots):** Interactive plots with filters and smoothing  
- **Page 3 (Extra):** Correlation analysis  
- **Page 4 (About):** Documentation and links  
""")

# Just to prove data is loaded (optional small preview)
st.write("### Preview of the dataset")
st.dataframe(df.head())
