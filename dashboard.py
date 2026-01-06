import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time
import os 

from db_manager import load_from_db
from etl_pipeline import run_pipeline

st.set_page_config(page_title="AI Financial Warehouse", layout="wide", page_icon="🏦")
st.title("🏦 AI Financial Forecasting Warehouse")

# AUTO-DETECT MODELS 
def get_available_tickers():
    """Scans the 'models/' folder and returns a list of trained tickers."""
    if not os.path.exists('models'):
        return []
    
    tickers = []
    for filename in os.listdir('models'):
        if filename.endswith("_cnn_lstm_v2.keras"):
            ticker_name = filename.split('_')[0].upper()
            tickers.append(ticker_name)
    return tickers

st.sidebar.header("Control Panel")

# Get list of valid models
valid_tickers = get_available_tickers()

# Safety Check
if not valid_tickers:
    st.sidebar.error("No Models Found!")
    st.warning("Please run 'forecaster.py' to train model.")
    st.stop()

ticker = st.sidebar.selectbox("Select Asset to Forecast", valid_tickers)

days_history = st.sidebar.slider("History to Plot (Days)", 100, 500, 200)

if st.sidebar.button(f"Run ETL for {ticker}"):
    with st.spinner("Updating Warehouse..."):
        run_pipeline() 
        time.sleep(1)
    st.success("Updated!")
    st.rerun()

# LOAD DATA 
def load_data(ticker_symbol):
    df = load_from_db(ticker_symbol)
    if df.empty: return None
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

df = load_data(ticker)
if df is None:
    st.error(f"No data found for {ticker} in Database. Did you run the ETL pipeline?")
    st.stop()

# PREPARE DATA 
features = ['close', 'SMA20', 'volume']
data_values = df[features].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data_values[:, 0].reshape(-1, 1))

# Load correct model based on selection
model_path = f'models/{ticker.lower()}_cnn_lstm_v2.keras'
model = load_model(model_path)

# BATCH PREDICTION
past_data = scaled_data[-days_history-60:] 

x_history = []
for i in range(60, len(past_data)):
    x_history.append(past_data[i-60:i, :])

x_history = np.array(x_history)

predicted_history = model.predict(x_history)
predicted_history_prices = close_scaler.inverse_transform(predicted_history)

graph_dates = df.index[-len(predicted_history_prices):]

# PREDICT 
last_60 = scaled_data[-60:].reshape(1, 60, 3)
next_day_scaled = model.predict(last_60)
next_day_price = close_scaler.inverse_transform(next_day_scaled)[0][0]

# VISUALIZE 
st.subheader(f"AI Performance Verification: {ticker}")

fig = go.Figure()

# Actual Price
fig.add_trace(go.Scatter(
    x=graph_dates, 
    y=df['close'].tail(len(graph_dates)),
    mode='lines', 
    name='Actual Price',
    line=dict(color='#00F0FF', width=2)
))

# Prediction
fig.add_trace(go.Scatter(
    x=graph_dates, 
    y=predicted_history_prices.flatten(),
    mode='lines', 
    name='AI Prediction (Backtest)',
    line=dict(color='#00FF00', width=2)
))

# Next Day
next_date = df.index[-1] + pd.Timedelta(days=1)
fig.add_trace(go.Scatter(
    x=[next_date], 
    y=[next_day_price],
    mode='markers+text', 
    name='Tomorrow Forecast',
    marker=dict(color='yellow', size=14, symbol='star'),
    text=[f"${next_day_price:.2f}"], 
    textposition="top center"
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Metrics
delta = next_day_price - df['close'].iloc[-1]
col1, col2 = st.columns(2)
col1.metric("Tomorrow's Prediction", f"${next_day_price:.2f}", f"{delta:.2f}")
col2.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")