# streamlit_app/Home.py
import streamlit as st

st.set_page_config(page_title="IND320 Project – Part 3", layout="wide")

st.title("IND320 – Data to Decision (Part 3)")
st.caption("NMBU / ESILV — Data Quality, Open-Meteo (ERA5), Elhub STL & Spectrogram, SPC/LOF")

# --- Global context preview (if set by page 2) ---
area = st.session_state.get("price_area")
if area:
    st.info(f"Current price area: **{area}**. Change it on the 'Production Analysis' page.")

st.markdown("""
### What’s inside
- **Page 2 – Production Analysis (global selector)**: choose the **price area** once; other pages reuse it.
- **Page 3 – STL & Spectrogram (new A)**: STL decomposition and time–frequency analysis on **Elhub** data.
- **Page 6 – Outliers & Anomalies (new B)**: **Temperature SPC (DCT)** and **Precipitation LOF** from **Open-Meteo ERA5**.
- Plus: table/plots/extra/about inherited from earlier parts.

### Notes
- Meteorology now comes from the **Open-Meteo ERA5 archive API** (no CSV).
- Elhub data are read from **MongoDB** (UTC timestamps).
""")

with st.expander("Links"):
    st.markdown("""
- GitHub repo: **JulesSylMUAMBA/IND320_DataTo_Decision** (branch `part3`)
- Streamlit pages order (requested): **1, 4, new A, 2, 3, new B, 5**  
  Here mapped to files:
  - `1_Table.py`
  - `2_Production_Analysis.py`  ← *(global area selector)*
  - `3_STL_Spectrogram.py`      ← *(new A)*
  - `4_Plots.py`
  - `5_Extra.py`
  - `6_Outliers_Anomalies.py`   ← *(new B)*
  - `7_About.py`
""")

st.success("Use the left sidebar to navigate between pages.")
