import streamlit as st

st.set_page_config(page_title="Outliers & Anomalies", page_icon="🧪", layout="wide")
st.title("New B — Outlier/SPC & Anomaly/LOF")
st.caption("This page will show Temperature SPC and Precipitation LOF in two tabs.")

tab1, tab2 = st.tabs(["Outlier / SPC (Temp)", "Anomaly / LOF (Precip)"])
with tab1:
    st.info("SPC tab placeholder — will display DCT+SPC outlier detection for temperature.")
with tab2:
    st.info("LOF tab placeholder — will display LOF anomalies for precipitation.")
