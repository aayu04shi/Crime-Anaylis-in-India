import pandas as pd

# Load leaderboard
df = pd.read_csv("leaderboard.csv")

# Sort by accuracy
df = df.sort_values(by="Accuracy", ascending=False)

# Add rank
df["Rank"] = range(1, len(df) + 1)

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
    table += f"| {row['Rank']} | {row['🏅']} | {row['GitHub']} | {row['Model']} | {row['Accuracy']:.2f}% |\n"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

start = "<!-- LEADERBOARD START -->"
end = "<!-- LEADERBOARD END -->"

# Replace content between markers
import re

pattern = re.compile(f"{start}.*?{end}", re.DOTALL)

new_section = f"{start}\n{table}\n{end}"

new_content = re.sub(pattern, new_section, content)

# Write back
with open("README.md", "w", encoding="utf-8") as f:
    f.write(new_content)

print("README updated successfully!")