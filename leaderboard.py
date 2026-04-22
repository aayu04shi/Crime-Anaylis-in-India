import streamlit as st
import pandas as pd
import os

st.title("🏆 GitHub Leaderboard")

file_path = "leaderboard.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    df = df.sort_values(by="Accuracy", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    st.dataframe(df, use_container_width=True)

else:
    st.warning("No leaderboard data yet.")
