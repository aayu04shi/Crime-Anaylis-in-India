import streamlit as st
import pandas as pd
import joblib
import os

from train_model import train_model # type: ignore

st.set_page_config(page_title="Crime Analysis", layout="centered")

st.title("🚔 Crime Pattern Analysis in India")

# ================= TRAIN MODEL BUTTON =================
github_user = st.text_input("Enter your GitHub Username")

if st.button("Train / Retrain Model"):
    

        df = pd.read_csv("data/crime_dataset_india.csv")
        model, X_test, y_test = train_model(df)

        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model trained! Accuracy: {round(acc*100, 2)}%")

        # ================= SAVE TO LEADERBOARD =================
        file_path = "leaderboard.csv"

if github_user.strip() != "": # type: ignore
    new_entry = pd.DataFrame(
        [[github_user, "RandomForest", acc]], # type: ignore
        columns=["GitHub", "Model", "Accuracy"]
    )

    if os.path.exists(file_path):
        old = pd.read_csv(file_path)

        # Remove duplicate entries (keep best score)
        old = old[old["GitHub"] != github_user] # type: ignore

        updated = pd.concat([old, new_entry], ignore_index=True)
    else:
        updated = new_entry

    updated = updated.sort_values(by="Accuracy", ascending=False)

    updated.to_csv(file_path, index=False)

    st.success("🏆 Score submitted to leaderboard!")
else:
    st.warning("Enter GitHub username!")


# ---------------- LOAD OR TRAIN ----------------
if not os.path.exists("models/trained_model.pkl"):

    st.warning("⚠️ Model not found. Training automatically...")

    df = pd.read_csv("data/crime_dataset_india.csv")
    model, X_test, y_test = train_model(df)

    st.success("✅ Model trained automatically!")

else:
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

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Align columns
        for col in model_columns:
            if col not in X_test:
                X_test[col] = 0
        X_test = X_test[model_columns]

        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

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
