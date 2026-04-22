import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Leaderboard", layout="centered")

st.title("🏆 Model Leaderboard")

file_path = "leaderboard.csv"

# ================= LOAD DATA =================
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Sort by accuracy (highest first)
    df = df.sort_values(by="Accuracy", ascending=False)

    # Reset ranking
    df["Rank"] = range(1, len(df) + 1)

    # Reorder columns
    df = df[["Rank", "Name", "Model", "Accuracy"]]

    st.dataframe(df, use_container_width=True)

else:
    st.warning("⚠️ No leaderboard data available yet.")
