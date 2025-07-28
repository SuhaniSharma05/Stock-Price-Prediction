# Stock Price Prediction with LSTM using Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Stock Price Prediction with LSTM", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GAIL.NS)", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
predict_days = st.sidebar.slider("Forecast Days", 30, 365, 30)

if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching data and training model..."):
        # Load data
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Close']].dropna()
        data = df.values

        # Normalize
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Train-Test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build LSTM
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=0)

        # Predict and Evaluate
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"*MAE*: {mae:.2f}")
        st.write(f"*RMSE*: {rmse:.2f}")
        st.write(f"*R² Score*: {r2:.2f}")

        # Plot test predictions
        st.subheader("📉 Actual vs Predicted (Test Data)")
        fig1, ax1 = plt.subplots()
        ax1.plot(y_test_inv, label='Actual')
        ax1.plot(y_pred_inv, label='Predicted')
        ax1.set_title("Actual vs Predicted")
        ax1.legend()
        st.pyplot(fig1)

        # Forecast next N days
        last_seq = scaled_data[-60:]
        future_preds = []

        for _ in range(predict_days):
            seq_input = last_seq.reshape(1, 60, 1)
            next_pred = model.predict(seq_input)[0][0]
            future_preds.append(next_pred)
            last_seq = np.append(last_seq[1:], [[next_pred]], axis=0)

        future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # Plot forecast
        st.subheader(f"🔮 Forecast for Next {predict_days} Days")
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds_inv.flatten()})

        fig2, ax2 = plt.subplots()
        ax2.plot(df.index[-100:], df['Close'].values[-100:], label='Recent Prices')
        ax2.plot(forecast_df['Date'], forecast_df['Predicted Close'], label='Forecast')
        ax2.set_title(f"{ticker} Price Forecast")
        ax2.legend()
        st.pyplot(fig2)

        # Show forecast table
        st.dataframe(forecast_df.set_index('Date').style.format({'Predicted Close': '{:.2f}'}))


