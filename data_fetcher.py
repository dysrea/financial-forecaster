import yfinance as yf
import pandas as pd
import os

def download_data(ticker="NVDA", start="2015-01-01", end="2026-01-05"):
    print(f"📡 Fetching data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    file_path = f'data/{ticker}_data.csv'
    data.to_csv(file_path)
    print(f"✅ Data saved to {file_path}")
    return data

if __name__ == "__main__":
    download_data()