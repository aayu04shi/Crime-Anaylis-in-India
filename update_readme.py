import pandas as pd

# Load leaderboard
df = pd.read_csv("leaderboard.csv")

def get_medal(rank):
    if rank == 1:
        return "🥇"
    elif rank == 2:
        return "🥈"
    elif rank == 3:
        return "🥉"
    else:
        return ""

df["Rank"] = df.index + 1
df["🏅"] = df["Rank"].apply(get_medal)
# Sort
df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

# Add rank
df["Rank"] = df.index + 1

# Convert accuracy to %
df["Accuracy"] = (df["Accuracy"] * 100).round(2).astype(str) + "%"

# Limit top 5
df = df.head(5)

# Create markdown table
table = "| Rank | GitHub | Model | Accuracy |\n"
table += "|------|--------|--------|----------|\n"

rows = ""
for _, row in df.iterrows():
    rows += f"| {row['🏅']} | {row['Rank']} | {row['GitHub']} | {row['Model']} | {round(row['Accuracy']*100,2)}% |\n"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# Replace section
start = "<!-- LEADERBOARD_START -->"
end = "<!-- LEADERBOARD_END -->"

new_content = content.split(start)[0] + start + "\n" + table + end + content.split(end)[1]

# Write back
with open("README.md", "w", encoding="utf-8") as f:
    f.write(new_content)

print("README updated successfully!")
