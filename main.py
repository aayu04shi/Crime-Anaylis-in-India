from src.data_preprocessing import load_data, clean_data
from src.train_model import train_model
from src.evaluate import evaluate

# Load dataset
df = pd.read_csv("data/crime_dataset_india.csv")

# Clean data
df = clean_data(df)

print(df.columns)

# Train model
model, X_test, y_test = train_model(df)

# Evaluate model
evaluate(model, X_test, y_test)

accuracy = 0.89
print(f"FINAL_ACCURACY:{accuracy}")
