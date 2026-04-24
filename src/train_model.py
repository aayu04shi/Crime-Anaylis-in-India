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

    df["Age_Group"] = pd.cut(
    df["Victim Age"],
    bins=[0, 18, 40, 60, 100],
    labels=[0, 1, 2, 3]
)

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
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["Crime Domain"])
    X = df.drop("Crime Domain", axis=1)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODEL ----------------
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


    model.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------
    evaluate(model, X_test, y_test)

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/trained_model.pkl", compress=3)
    joblib.dump(X.columns, "models/columns.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("✅ Model and columns saved successfully!")

    return model, X_test, y_test
