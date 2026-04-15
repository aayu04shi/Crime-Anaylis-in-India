import streamlit as st
import pandas as pd
import pickle

st.title("🚔 Crime Pattern Analysis")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    st.write("Columns:", df.columns)

    try:
        model = pickle.load(open("models/trained_model.pkl", "rb"))

        st.subheader("Prediction")

        input_data = []
        for col in df.columns[:-1]:
            val = st.number_input(f"{col}", value=0)
            input_data.append(val)

        if st.button("Predict"):
            result = model.predict([input_data])
            st.success(result)

    except:
        st.warning("Model not trained yet")
