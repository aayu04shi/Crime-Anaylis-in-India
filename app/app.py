import streamlit as st
import pandas as pd
import joblib

st.title("🚔 Crime Pattern Analysis in India")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("models/trained_model.pkl")
except:
    st.error("Model not found or corrupted. Please download it again.")
    st.stop()

# ================= PREDICTION SECTION (TOP) =================
st.subheader("Enter Crime Details")

city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Pune"])
victim_age = st.number_input("Victim Age", min_value=0, max_value=100)
gender = st.selectbox("Victim Gender", ["Male", "Female"])
weapon = st.selectbox("Weapon Used", ["Knife", "Gun", "None", "Unknown"])
police = st.number_input("Police Deployed", min_value=0)

# Create dataframe
input_data = pd.DataFrame({
    "City": [city],
    "Victim Age": [victim_age],
    "Victim Gender": [gender],
    "Weapon Used": [weapon],
    "Police Deployed": [police]
})

# Encode
input_data = pd.get_dummies(input_data)

# Align with model features
model_columns = model.feature_names_in_

for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[model_columns]

# Prediction button
if st.button("Predict Crime Type"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Crime Domain: {prediction[0]}")

# ================= DATASET SECTION (BOTTOM) =================
st.subheader("📊 Dataset Analysis")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.write(df.head())

    if "Crime Domain" in df.columns:
        st.write("### Crime Distribution")
        st.bar_chart(df["Crime Domain"].value_counts())
