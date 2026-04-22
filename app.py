from supabase import create_client

url = "https://rlbhqkhrbapxetjoubzm.supabase.co/rest/v1/"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJsYmhxa2hyYmFweGV0am91YnptIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY4NzUyOTgsImV4cCI6MjA5MjQ1MTI5OH0.YmYLnl3uvOl8DHkroRAaeT011VueTivS_U2V55xSDLs"

supabase = create_client(url, key)
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crime Analysis", layout="centered")

st.title("🚔 Crime Pattern Analysis in India")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("models/trained_model.pkl")
    model_columns = joblib.load("models/columns.pkl")
except:
    st.error("❌ Model or columns file not found. Please train the model first.")
    st.stop()

# ================= PREDICTION SECTION =================
st.subheader("🔍 Enter Crime Details")

city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Pune"])
victim_age = st.number_input("Victim Age", min_value=0, max_value=100)
gender = st.selectbox("Victim Gender", ["Male", "Female"])
weapon = st.selectbox("Weapon Used", ["Knife", "Gun", "None", "Unknown"])
police = st.number_input("Police Deployed", min_value=0)

# Create input dataframe
input_data = pd.DataFrame({
    "City": [city],
    "Victim Age": [victim_age],
    "Victim Gender": [gender],
    "Weapon Used": [weapon],
    "Police Deployed": [police]
})

# ---------------- ENCODING ----------------
input_data = pd.get_dummies(input_data)

# Add missing columns
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

# Ensure correct order
input_data = input_data[model_columns]

# ---------------- PREDICTION ----------------
if st.button("Predict Crime Type"):
    prediction = model.predict(input_data)
    st.success(f"🚨 Predicted Crime Domain: {prediction[0]}")

# ================= DATASET SECTION =================
st.subheader("📊 Dataset Analysis")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    if "Crime Domain" in df.columns:
        st.write("### Crime Distribution")
        st.bar_chart(df["Crime Domain"].value_counts())
