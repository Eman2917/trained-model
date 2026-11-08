import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Water Potability Checker", page_icon="💧", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #e0f2fe, #f0fdfa); }
.main-header { background: linear-gradient(90deg, #1e3a8a, #06b6d4); padding: 25px; border-radius: 15px; color: white; text-align: center; }
.stButton>button { background: linear-gradient(90deg, #06b6d4, #0891b2); color: white; font-size: 20px; font-weight: bold; border-radius: 30px; height: 55px; width: 100%; border: none; }
</style>
""", unsafe_allow_html=True)

model = joblib.load('water_model.pkl')
imputer = joblib.load('imputer.pkl')

st.markdown('<div class="main-header"><h1>💧 Water Potability Checker</h1><p>Slider ghumao, value set karo, CHECK dabao</p></div>', unsafe_allow_html=True)
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.slider("🧪 pH Level", 0.0, 14.0, 7.0)
    chloramines = st.slider("🧪 Chloramines", 0.0, 15.0, 7.0)
    organic_carbon = st.slider("🌱 Organic Carbon", 0.0, 30.0, 14.0)

with col2:
    hardness = st.slider("🪨 Hardness", 0.0, 350.0, 200.0)
    sulfate = st.slider("🧂 Sulfate", 0.0, 500.0, 300.0)
    trihalomethanes = st.slider("⚗️ Trihalomethanes", 0.0, 150.0, 60.0)

with col3:
    solids = st.slider("💎 Solids", 0.0, 60000.0, 20000.0)
    conductivity = st.slider("⚡ Conductivity", 0.0, 1000.0, 400.0)
    turbidity = st.slider("🌫️ Turbidity", 0.0, 10.0, 4.0)

if st.button("🔍 PREDICT POTABILITY"):
    inputs = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    inputs = imputer.transform(inputs)
    result = model.predict(inputs)
    prob = model.predict_proba(inputs)[0][1] * 100

    if result[0] == 1:
        st.success(f"✅ Water Is Potable | SAFE: {prob:.1f}%")
    else:
        st.error(f"❌ Water Is Not Potable | NOT SAFE: {prob:.1f}%")
