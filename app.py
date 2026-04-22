import pandas as pd
import os
import streamlit as st
import pandas as pd
import joblib
import os

from train_model import train_model # type: ignore

st.set_page_config(page_title="Crime Analysis", layout="centered")

st.title("🚔 Crime Pattern Analysis in India")

# ================= TRAIN MODEL BUTTON =================
st.subheader("⚙️ Train Model")

if st.button("Train / Retrain Model"):
    try:
        df = pd.read_csv("data/crime_dataset_india.csv")
        model, X_test, y_test = train_model(df)

        # Calculate accuracy again for UI
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model trained successfully! Accuracy: {round(acc*100, 2)}%")

    except Exception as e:
        st.error(f"❌ Error during training: {e}")

# ================= LOAD MODEL =================
model = None
model_columns = None

if os.path.exists("models/trained_model.pkl") and os.path.exists("models/columns.pkl"):
    model = joblib.load("models/trained_model.pkl")
    model_columns = joblib.load("models/columns.pkl")

# ================= SHOW ACCURACY =================
st.subheader("📊 Model Performance")

if model is not None:
    try:
        df = pd.read_csv("data/crime_dataset_india.csv")

        # Prepare data same as training
        from data_preprocessing import clean_data # type: ignore
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
        acc = accuracy_score(y, y_pred)

        st.info(f"📈 Current Model Accuracy: {round(acc*100, 2)}%")

    except:
        st.warning("⚠️ Could not calculate accuracy")

else:
    st.warning("⚠️ Model not found. Please train the model first.")

# ================= PREDICTION =================
st.subheader("🔍 Predict Crime Type")

if model is not None:

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

    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_data = input_data[model_columns]

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"🚨 Predicted Crime Domain: {prediction[0]}")

else:
    st.info("👉 Train the model first to enable prediction")

# ================= DATASET VIEW =================
st.subheader("📂 Dataset Preview")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if "Crime Domain" in df.columns:
        st.bar_chart(df["Crime Domain"].value_counts())
