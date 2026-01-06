import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --- NEW: Import from your Database Manager ---
from db_manager import load_from_db

# 1. Load Data from SQL (No more CSVs!)
print("🔌 Querying Database for Visualization...")
df = load_from_db('NVDA')

# Safety Check
if df.empty:
    print("❌ Error: Database is empty!")
    exit()

# 2. Preprocess (Match the lowercase SQL columns)
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

df['SMA20'] = df['close'].rolling(window=20).mean()
df.dropna(inplace=True)

features = ['close', 'SMA20', 'volume']
data_values = df[features].values

# 3. Load Model & Scale (Same as before)
model = load_model('models/nvda_cnn_lstm_v2.keras')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data_values[:, 0].reshape(-1, 1))

# 4. Prepare Test Data
test_inputs = scaled_data[len(scaled_data) - 100 - 60:]
x_test = []
for i in range(60, len(test_inputs)):
    x_test.append(test_inputs[i-60:i, :])
x_test = np.array(x_test)

# 5. Predict
predicted_prices = model.predict(x_test)
predicted_prices = close_scaler.inverse_transform(predicted_prices)
real_prices = data_values[len(data_values) - 100:, 0]

# 6. Plot
plt.figure(figsize=(14, 7))
plt.plot(real_prices, color='black', label='Actual Price (Source: SQLite)', linewidth=2)
plt.plot(predicted_prices, color='green', label='AI Prediction', linewidth=2)
plt.title('NVIDIA Forecast: Powered by SQL Data Warehouse')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()