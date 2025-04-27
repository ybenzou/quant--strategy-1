import os
import pandas as pd
import numpy as np
import yfinance as yf

def download_stock_data(tickers, start_date, end_date, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    for ticker in tickers:
        print(f"[INFO] Downloading {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            output_path = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(output_path)
            print(f"[INFO] Saved {ticker} data to {output_path}")

def compute_features(df):
    df = df.copy()

    # 基础特征
    df["Return"] = df["Close"].pct_change()
    df["Volume_Change"] = df["Volume"].pct_change()

    # 均线特征
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA5_minus_MA20"] = df["MA5"] - df["MA20"]

    # 波动率
    df["Volatility10"] = df["Close"].pct_change().rolling(window=10).std()

    # RSI特征
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    return df

def load_and_prepare_data(tickers, start_date, end_date, data_dir="data"):
    final_data = {}
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            print(f"[WARN] {file_path} not found, downloading...")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            data.to_csv(file_path)
            data.index.name = "Date"

        # 读取数据，关键：跳过第二行Ticker行
        df = pd.read_csv(file_path, skiprows=[1])

        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)

        # 关键列转成数值
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 必须有Close列
        if "Close" not in df.columns:
            print(f"[WARN] {ticker} missing Close column, skipped")
            continue

        df = compute_features(df)
        final_data[ticker] = df

    return final_data
