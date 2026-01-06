import yfinance as yf
import pandas as pd
from db_manager import save_to_db

TICKER = "NVDA"
START_DATE = "2022-01-01"

def run_pipeline():
    print(f"Starting ETL Pipeline for {TICKER}...")
    
    # EXTRACT
    print("Extracting data from Yahoo Finance API...")
    df = yf.download(TICKER, start=START_DATE, progress=False)
    
    if len(df) == 0:
        print("Error: No data fetched. Check your internet or ticker.")
        return

    # TRANSFORM CHECK
    print(f"Transformed {len(df)} rows of raw data.")
    
    # LOAD
    print("Loading data into SQLite Warehouse...")
    save_to_db(df, TICKER)
    
    print("ETL Job Complete. Data is safely stored in SQL.")

if __name__ == "__main__":
    run_pipeline()