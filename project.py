import os

# 定义你的项目结构
project_structure = {
    "strategy1": [
        "README.md",
        "requirements.txt",
        "projectStructure.py",
        "main.py",
        "backtest/backtester.py",
        "data/",
        "data_fetcher/fetcher.py",
        "notebooks/",
        "plots/",
        "scripts/data/AAPL.csv",
        "strategies/moving_average.py",
        "utils/visualization.py",
    ]
}

# 填充一些初始内容（比如README, main.py）
file_templates = {
    "README.md": "# Strategy1: Stock Quantitative Model Project\n",
    "requirements.txt": "yfinance\npandas\nmatplotlib\n",
    "main.py": '''from data_fetcher.fetcher import fetch_stock_data
from utils.visualization import plot_stock_price_and_volume

def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    data = fetch_stock_data(ticker, start_date, end_date)
    plot_stock_price_and_volume(data, ticker)

if __name__ == "__main__":
    main()
''',
    "data_fetcher/fetcher.py": '''import os
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
''',
    "utils/visualization.py": '''import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_price_and_volume(data: pd.DataFrame, ticker: str):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(data.index, data["Close"], color='blue', label="Close Price")
    ax1.set_ylabel('Price', fontsize=14)
    ax1.set_title(f"{ticker} Closing Price and Volume", fontsize=16)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(data.index, data["Volume"], color='gray', alpha=0.3, label="Volume")
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc="upper right")

    plt.grid(True)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{ticker}_overview.png"
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to {plot_path}")

    plt.show()
''',
    "strategies/moving_average.py": '''# Moving average strategy placeholder

class MovingAverageStrategy:
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        data["ShortMA"] = data["Close"].rolling(window=self.short_window, min_periods=1).mean()
        data["LongMA"] = data["Close"].rolling(window=self.long_window, min_periods=1).mean()
        data["Signal"] = 0
        data.loc[data["ShortMA"] > data["LongMA"], "Signal"] = 1
        data.loc[data["ShortMA"] <= data["LongMA"], "Signal"] = -1
        return data
''',
    "backtest/backtester.py": '''# Backtester placeholder

class Backtester:
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals

    def run_backtest(self):
        returns = self.data["Close"].pct_change()
        strategy_returns = returns * self.signals.shift(1)
        cumulative_returns = (1 + strategy_returns).cumprod()
        return cumulative_returns
'''
}

def create_project():
    for root, files in project_structure.items():
        for file in files:
            file_path = os.path.join(root, file)

            # 创建目录
            dir_path = file_path if file.endswith("/") else os.path.dirname(file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"[DIR] Created directory: {dir_path}")

            # 创建文件（如果不是目录）
            if not file.endswith("/"):
                if not os.path.exists(file_path):
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(file_templates.get(file, ""))
                    print(f"[FILE] Created file: {file_path}")

if __name__ == "__main__":
    create_project()
