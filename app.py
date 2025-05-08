
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.title("Stock Price Prediction with LSTM")

uploaded_file = st.file_uploader("Upload your stock dataset (CSV with 'Close' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df.head())

    if 'Close' not in df.columns:
        st.error("CSV must contain a 'Close' column.")
    else:
        close_data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        seq_length = 60
        X_test = []
        for i in range(seq_length, len(scaled_data)):
            X_test.append(scaled_data[i-seq_length:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        try:
            model = load_model("lstm_model.h5")
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)

            actual = close_data[seq_length:]
            result_df = pd.DataFrame({
                "Actual": actual.flatten(),
                "Predicted": predictions.flatten()
            })

            st.line_chart(result_df)
        except Exception as e:
            st.error(f"Error loading model: {e}")
