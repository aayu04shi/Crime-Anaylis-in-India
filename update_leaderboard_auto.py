import pandas as pd
import os

SUBMISSIONS_FOLDER = "submissions"
LEADERBOARD_FILE = "leaderboard.csv"

all_data = []

# Read all submission files
if os.path.exists(SUBMISSIONS_FOLDER):
    for file in os.listdir(SUBMISSIONS_FOLDER):
        if file.endswith(".csv"):
            path = os.path.join(SUBMISSIONS_FOLDER, file)
            df = pd.read_csv(path)
            all_data.append(df)

# Combine all
if all_data:
    combined = pd.concat(all_data, ignore_index=True)

    # Keep best score per user
    combined = combined.sort_values(by="Accuracy", ascending=False)
    leaderboard = combined.drop_duplicates(subset="GitHub", keep="first")

    # Sort again
    leaderboard = leaderboard.sort_values(by="Accuracy", ascending=False)

    leaderboard.to_csv(LEADERBOARD_FILE, index=False)

    print("✅ Leaderboard updated successfully!")

else:
    print("⚠️ No submissions found.")
