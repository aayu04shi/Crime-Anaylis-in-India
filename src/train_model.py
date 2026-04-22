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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # One-hot encoding after split to avoid leakage from test data
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

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
    joblib.dump(X_train.columns, "models/columns.pkl")

    print("✅ Model and columns saved successfully!")

    return model, X_test, y_test
