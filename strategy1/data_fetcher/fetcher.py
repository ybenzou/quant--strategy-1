import os
import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker: str, start_date: str, end_date: str, output_dir: str = "scripts/data") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}.csv")

    if os.path.exists(output_path):
        print(f"[INFO] Loading cached data for {ticker}")
        data = pd.read_csv(output_path, index_col="Date", parse_dates=True)
    else:
        print(f"[INFO] Downloading data for {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        data.to_csv(output_path)
        print(f"[INFO] Data for {ticker} saved to {output_path}")
        data = pd.read_csv(output_path, index_col="Date", parse_dates=True)

    return data
