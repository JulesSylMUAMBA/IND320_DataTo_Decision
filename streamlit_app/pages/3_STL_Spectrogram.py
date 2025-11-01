import streamlit as st

st.set_page_config(page_title="STL & Spectrogram", page_icon="📈", layout="wide")
st.title("New A — STL decomposition & Spectrogram")
st.caption("This page will show STL (Elhub) and a Spectrogram (Elhub) in two tabs.")

tab1, tab2 = st.tabs(["STL", "Spectrogram"])
with tab1:
    st.info("STL tab placeholder — will display STL decomposition of Elhub data.")
with tab2:
    st.info("Spectrogram tab placeholder — will display time–frequency analysis of Elhub data.")
