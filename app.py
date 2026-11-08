import streamlit as st
import joblib
import numpy as np

model = joblib.load('water_model.pkl')
imputer = joblib.load('imputer.pkl')

st.title("💧 Water Potability Checker")

ph = st.number_input("pH", value=7.0)
hardness = st.number_input("Hardness", value=200.0)
solids = st.number_input("Solids", value=20000.0)
chloramines = st.number_input("Chloramines", value=7.0)
sulfate = st.number_input("Sulfate", value=300.0)
conductivity = st.number_input("Conductivity", value=400.0)
organic_carbon = st.number_input("Organic Carbon", value=14.0)
trihalomethanes = st.number_input("Trihalomethanes", value=60.0)
turbidity = st.number_input("Turbidity", value=4.0)

if st.button("Check Karo"):
    inputs = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    inputs = imputer.transform(inputs)
    result = model.predict(inputs)
    if result[0] == 1:
        st.success("ok")
    else:
        st.error("not ok")