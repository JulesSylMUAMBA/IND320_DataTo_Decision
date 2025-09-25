import streamlit as st

st.title("About this App")

st.write("""
This Streamlit app was developed as part of the **IND320 – Data to Decision** course.  
It demonstrates how to:
- Load and explore weather data (`open-meteo-subset.csv`)
- Display variables in a table with mini time series (Page 1)
- Interactively plot selected variables with filters (Page 2)
- Analyze correlations between variables (Page 3)

---

### Author
- **Name:** Jules Syl Muamba  
- **Institution:** NMBU / ESILV Exchange  
- **Course:** IND320 – Data to Decision  

---

### Links
- **GitHub Repository:** [IND320_DataTo_Decision](https://github.com/JulesSylMUAMBA/IND320_DataTo_Decision)  
- **Streamlit App:** [share.streamlit.io/user/julessylmuamba](https://share.streamlit.io/user/julessylmuamba)  

---

### Notes
The app uses:
- `pandas` for data handling  
- `matplotlib` for visualization  
- Streamlit’s `LineChartColumn` for mini-plots in tables  
- `@st.cache_data` to improve performance when loading the dataset  

""")
