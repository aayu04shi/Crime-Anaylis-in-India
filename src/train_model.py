import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import clean_data # type: ignore
from src.evaluate import evaluate # type: ignore


def train_model(df):

    # ---------------- PREPROCESS ----------------
    df = clean_data(df)

    # Drop unnecessary columns
    df = df.drop([
        "Report Number",
        "Date Reported",
        "Date of Occurrence",
        "Time of Occurrence",
        "Crime Description",
        "Date Case Closed"
    ], axis=1)

    # ---------------- SPLIT ----------------
    y = df["Crime Domain"]
    X = df.drop("Crime Domain", axis=1)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODEL ----------------
    model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
    model.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------
    evaluate(model, X_test, y_test)

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/trained_model.pkl", compress=3)
    joblib.dump(X.columns, "models/columns.pkl")

    print("✅ Model and columns saved successfully!")

    return model, X_test, y_test
