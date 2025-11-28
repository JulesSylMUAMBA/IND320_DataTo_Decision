import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About this Application")
st.markdown("""
This dashboard was developed as part of the **IND320 – Data to Decision** course  
at **NMBU**, during an academic exchange with **ESILV Paris**.

The goal of the project is to combine **energy data (Elhub)**,  
**meteorological reanalysis (ERA5 – Open-Meteo)**, and **geospatial layers**  
to build an interactive analytical tool covering:

---

## 🔍 Main Features

### **1. Data Exploration**
- Table-based exploration of clean weather datasets  
- Interactive Plotly charts  
- Monthly and multi-variable visualization  
- Global price area selection reused across pages

### **2. Advanced Time-Series Analysis**
- **STL decomposition** (trend, seasonality, residuals)  
- **Spectrogram** (time–frequency structure of hourly energy production)

### **3. Anomaly Detection**
- **SPC on temperature (DCT filtering)**  
- **LOF detection on precipitation**  
- Dynamic controls for robustness, sigma, contamination, neighbors…  
- Outliers visualized directly on interactive plots

### **4. Geospatial Visualization**
- Folium-based map of **NO1–NO5** (Elspot areas) with:
  - Choropleth coloring based on production/consumption
  - Interactive click → store coordinates
  - Highlight selected area
- Coordinates are reused in the Snow Drift analysis

### **5. Snow Drift & Wind Rose**
- ERA5-based hydrological-year snow drift calculation  
- Multi-year comparison  
- **Wind rose** built from wind speed & direction  
- Fully dynamic with user-selected map coordinates

### **6. Predictive Modelling**
- Sliding-window correlation between energy and weather variables  
- **SARIMAX forecasting** with:
  - Selectable p, d, q, P, D, Q, season length  
  - Selectable training period  
  - Support for exogenous meteorological variables  
  - Confidence interval plotting

---

## 👨‍💻 Author
**Jules Sylvain Muamba Mvele**  
ESILV – NMBU Exchange Student  
IND320 – Data to Decision

---

## 📎 Useful Links
- **GitHub Repository:**  
  https://github.com/JulesSylMUAMBA/IND320_DataTo_Decision

- **Streamlit App:**  
  https://ind320datatodecision-fnmdxfu8zeflxwwdgdjvmx.streamlit.app/
            

---

## 🛠 Technologies Used
- **Streamlit** for the application framework  
- **Pandas** for data processing  
- **Plotly** for interactive graphics  
- **Folium** for geospatial visualization  
- **MongoDB** for Elhub storage  
- **Open-Meteo ERA5 API** for meteorology  
- **Statsmodels** for SARIMAX  
- **scikit-learn** for anomaly detection (LOF)  

---

## 📘 Notes
This application was designed to follow the requirements of Part 4 of the IND320 project:
- Exploratory → Anomalies → Geospatial → Snow Drift → Predictive Modelling  
- Smooth navigation with persistent selections (e.g., price area, coordinates)  
- All visualizations are **interactive** and optimized for classroom demonstration.

""")
