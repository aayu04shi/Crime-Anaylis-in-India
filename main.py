import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train, y_train) # Train model
        
        y_test_pred = model.predict(X_test)
        test_model_score = r2_score(y_test, y_test_pred)
        
        report[list(models.keys())[i]] = test_model_score
    return report

# The "Leaderboard" Logic
models = {
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso()
}

model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

# Automating the "Top Rank"
best_model_score = max(sorted(model_report.values()))
best_model_name = list(model_report.keys())[
    list(model_report.values()).index(best_model_score)
]
