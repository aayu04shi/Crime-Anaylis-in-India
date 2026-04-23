import pandas as pd
import os

submissions_folder = "submissions"
leaderboard_file = "leaderboard.csv"

all_data = []

# Read all submissions
for file in os.listdir(submissions_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(submissions_folder, file))
        all_data.append(df)

if len(all_data) == 0:
    print("No submissions found")
    exit()

# Combine all
df = pd.concat(all_data, ignore_index=True)

# Keep BEST score per user
df = df.sort_values("Accuracy", ascending=False)
df = df.drop_duplicates(subset=["GitHub"], keep="first")

# Rank
df = df.reset_index(drop=True)
df["Rank"] = df.index + 1

# Save leaderboard
df.to_csv(leaderboard_file, index=False)

print("Leaderboard updated!")
