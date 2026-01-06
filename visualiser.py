import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load data and model
df = pd.read_csv('data/NVDA_data.csv')
data = pd.to_numeric(df['Close'].values, errors='coerce')
data = data[~np.isnan(data)].reshape(-1, 1)

model = load_model('models/nvda_cnn_lstm.h5')

# Setup scaler, same as training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prep test data
test_inputs = scaled_data[len(scaled_data) - 100 - 60:] # Get last 100 days + window
x_test = []

for i in range(60, len(test_inputs)):
    x_test.append(test_inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices) # Back to Dollars

# Get real prices 
real_prices = data[len(data) - 100:]

# Plot
plt.figure(figsize=(12,6))
plt.plot(real_prices, color='black', label='Actual NVDA Price')
plt.plot(predicted_prices, color='green', label='LSTM Predicted Price')
plt.title('NVIDIA Stock Price Prediction')
plt.xlabel('Time (Last 100 Days)')
plt.ylabel('Price ($)')
plt.legend()
plt.show()