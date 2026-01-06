import sqlite3
import pandas as pd

DB_NAME = "financial_data.db"

def get_connection():
    """Connect to the local SQLite database."""
    return sqlite3.connect(DB_NAME)

def create_tables():
    """Create the robust schema for our stock data."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create table 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        ingestion_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date) ON CONFLICT REPLACE
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} and tables created successfully.")

def save_to_db(df, ticker):
    """
    ETL Function: Takes a DataFrame and loads it into SQL.
    """
    conn = get_connection()
    
    # If columns look like ('Close', 'NVDA'), flatten them to just 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make 'Date' a normal column
    df_clean = df.reset_index()
    
    # 2. Rename columns to match SQL schema
    df_clean = df_clean.rename(columns={
        "Date": "date", "Open": "open", "High": "high", 
        "Low": "low", "Close": "close", "Volume": "volume",
        "Adj Close": "adj_close" 
    })
    
    # 3. Add Ticker column
    df_clean['ticker'] = ticker
    
    # Ignore any extra columns/artifacts
    columns_to_keep = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    
    # Filter the dataframe safely
    available_cols = [c for c in columns_to_keep if c in df_clean.columns]
    df_clean = df_clean[available_cols]
    
    # Load
    try:
        df_clean.to_sql('stock_prices', conn, if_exists='append', index=False)
        print(f"Successfully saved {len(df_clean)} rows for {ticker} to SQL.")
    except Exception as e:
        print(f"Error saving to DB: {e}")
        
    conn.close()

def load_from_db(ticker):
    """
    Extract Function: specific query to get data for training.
    """
    conn = get_connection()
    
    query = f"""
    SELECT date, close, volume 
    FROM stock_prices 
    WHERE ticker = '{ticker}' 
    ORDER BY date ASC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Initialize DB when script is run directly
if __name__ == "__main__":
    create_tables()