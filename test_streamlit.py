import streamlit as st

st.header("Test Streamlit Application")


st.write("This is a test file for Streamlit.")

# Add more Streamlit components as needed for testing
st.button("Test Button")
st.text_input("Mets un truc ici")

# Add a button to display a message
if st.button("Click me!"):
    st.write("Button clicked!")