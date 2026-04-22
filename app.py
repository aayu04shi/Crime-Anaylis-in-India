import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import joblib
import os

from src.train_model import train_model
from src.data_preprocessing import clean_data

st.set_page_config(page_title="Crime Analysis", layout="centered")

st.title("🚔 Crime Pattern Analysis in India")

# ================= TRAIN MODEL + LEADERBOARD =================
st.subheader("⚙️ Train Model & Submit Score")

github_user = st.text_input("Enter your GitHub Username")

acc = None  # initialize

if st.button("Train / Retrain Model"):

    df = pd.read_csv("data/crime_dataset_india.csv")

    model, X_test, y_test = train_model(df)

    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model trained! Accuracy: {round(acc*100, 2)}%")

    # -------- SAVE TO LEADERBOARD --------
    if github_user.strip() != "":
        file_path = "leaderboard.csv"

        new_entry = pd.DataFrame(
            [[github_user, "RandomForest", acc]],
            columns=["GitHub", "Model", "Accuracy"]
        )

        if os.path.exists(file_path):
            old = pd.read_csv(file_path)

            # Remove duplicate entries
            old = old[old["GitHub"] != github_user]

            updated = pd.concat([old, new_entry], ignore_index=True)
        else:
            updated = new_entry

        updated = updated.sort_values(by="Accuracy", ascending=False)
        updated.to_csv(file_path, index=False)

        st.success("🏆 Score submitted to leaderboard!")

    else:
        st.warning("⚠️ Enter GitHub username!")

# ================= LOAD OR AUTO-TRAIN MODEL =================
st.subheader("🤖 Model Initialization")

if not os.path.exists("models/trained_model.pkl"):

    st.warning("⚠️ Model not found. Training automatically...")

    df = pd.read_csv("data/crime_dataset_india.csv")

    with st.spinner("Training model..."):
        model, X_test, y_test = train_model(df)

    model_columns = X_test.columns

    st.success("✅ Model trained automatically!")

else:
    model = joblib.load("models/trained_model.pkl")
    model_columns = joblib.load("models/columns.pkl")

# ================= MODEL PERFORMANCE =================
st.subheader("📊 Model Performance")

try:
    df = pd.read_csv("data/crime_dataset_india.csv")
    df = clean_data(df)

    df = df.drop([
        "Report Number",
        "Date Reported",
        "Date of Occurrence",
        "Time of Occurrence",
        "Crime Description",
        "Date Case Closed"
    ], axis=1)

    y = df["Crime Domain"]
    X = df.drop("Crime Domain", axis=1)

    X = pd.get_dummies(X, drop_first=True)

    # Align columns
    for col in model_columns:
        if col not in X:
            X[col] = 0

    X = X[model_columns]

    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X)
    acc_full = accuracy_score(y, y_pred)

    st.info(f"📈 Current Model Accuracy: {round(acc_full*100, 2)}%")

except Exception as e:
    st.warning("⚠️ Could not calculate accuracy")

# ================= PREDICTION =================
st.subheader("🔍 Predict Crime Type")

city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Pune"])
victim_age = st.number_input("Victim Age", min_value=0, max_value=100)
gender = st.selectbox("Victim Gender", ["Male", "Female"])
weapon = st.selectbox("Weapon Used", ["Knife", "Gun", "None", "Unknown"])
police = st.number_input("Police Deployed", min_value=0)

input_data = pd.DataFrame({
    "City": [city],
    "Victim Age": [victim_age],
    "Victim Gender": [gender],
    "Weapon Used": [weapon],
    "Police Deployed": [police]
})

input_data = pd.get_dummies(input_data)

# Align columns
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[model_columns]

if st.button("Predict Crime Type"):
    prediction = model.predict(input_data)
    st.success(f"🚨 Predicted Crime Domain: {prediction[0]}")

# ================= DATASET VIEW =================
st.subheader("📂 Dataset Preview")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if "Crime Domain" in df.columns:
        st.bar_chart(df["Crime Domain"].value_counts())
