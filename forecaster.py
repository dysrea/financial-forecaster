import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D

# Import the loader from your database manager
from db_manager import load_from_db 

# Load Data from SQL
print("Connecting to Database...")
df = load_from_db('NVDA') 

# Safety Check
if df.empty:
    print("Error: Database is empty. Did you run 'etl_pipeline.py'?")
    exit()

# Preprocessing 
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Calculate SMA
df['SMA20'] = df['close'].rolling(window=20).mean()
df.dropna(inplace=True)

# Select features
features = ['close', 'SMA20', 'volume']
data_values = df[features].values
print(f"Data loaded. Shape: {data_values.shape}")

# Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

# Sliding Window
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, :]) 
    y_train.append(scaled_data[x, 0]) # Predicting 'close' price (index 0)

x_train, y_train = np.array(x_train), np.array(y_train)

# Build Model
model = Sequential([
    Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.1),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
print("Training on SQL Data...")
model.fit(x_train, y_train, batch_size=16, epochs=50)

# Save
import os
if not os.path.exists('models'):
    os.makedirs('models')
    
model.save('models/nvda_cnn_lstm_v2.keras')
print("Success. Model trained on database and saved.")