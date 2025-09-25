# IND320 – Data to Decision

This repository contains my work for the IND320 course at NMBU.  
The project combines a Jupyter Notebook (for data exploration and documentation) and a Streamlit app (for interactive visualization).  

---

## Dataset
The dataset used is `open-meteo-subset.csv`, which contains:
- Time (timestamp)
- Temperature (°C)
- Precipitation (mm)
- Wind speed at 10m (m/s)
- Wind gusts at 10m (m/s)
- Wind direction at 10m (°)

---

## Jupyter Notebook
- Loads the dataset with **pandas**
- Plots each column separately with **matplotlib**
- Plots all columns together (with normalization and alternative views for readability)
- Includes a **correlation analysis**
- Contains a **300–500 word log** describing the work and use of AI tools
- Exported as PDF for submission

---

## Streamlit App
The Streamlit app is structured into four pages:

1. **Table** – One row per variable with a `LineChartColumn` mini-plot (first month).  
2. **Plots** – Interactive plots with:
   - A dropdown to select variables
   - A slider to select months
   - An optional smoothing (7-day rolling mean)  
3. **Extra** – Correlation matrix (heatmap + numeric table).  
4. **About** – Project description, author info, and links.  

---

## Links
- **GitHub Repository:** [IND320_DataTo_Decision](https://github.com/JulesSylMUAMBA/IND320_DataTo_Decision)  
- **Streamlit App:** [https://ind320datatodecision-fnmdxfu8zeflxwwdgdjvmx.streamlit.app/s)  

---

## Requirements
Main dependencies:
- `streamlit`
- `pandas`
- `matplotlib`

(see `requirements.txt` for details)

---

## Author
- **Name:** Jules Syl Muamba  
- **Institution:** NMBU / ESILV Exchange  
- **Course:** IND320 – Data to Decision
