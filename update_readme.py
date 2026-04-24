import pandas as pd

# Load leaderboard
df = pd.read_csv("leaderboard.csv")

# Sort properly
df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

# Add rank
df["Rank"] = df.index + 1

# Medal function
def get_medal(rank):
    if rank == 1:
        return "🥇"
    elif rank == 2:
        return "🥈"
    elif rank == 3:
        return "🥉"
    return ""

df["🏅"] = df["Rank"].apply(get_medal)

# Create markdown table
table = "| Rank | 🏅 | GitHub | Model | Accuracy |\n"
table += "|------|----|--------|--------|----------|\n"

for _, row in df.iterrows():
    table += f"| {row['Rank']} | {row['🏅']} | {row['GitHub']} | {row['Model']} | {round(row['Accuracy']*100, 2)}% |\n"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

start = "<!-- LEADERBOARD START -->"
end = "<!-- LEADERBOARD END -->"

# Replace section
new_content = content.split(start)[0] + start + "\n" + table + "\n" + end + content.split(end)[1]

# Write back
with open("README.md", "w", encoding="utf-8") as f:
    f.write(new_content)

print("README updated successfully!")
