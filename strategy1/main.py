from datetime import datetime
from data_processor.feature_engineering import download_stock_data, load_and_prepare_data
from model.signal_model import train_signal_model, predict_signals
import pandas as pd
import os

def save_trading_signals(signals_dict):
    """
    保存今天的交易信号到历史文件，包括信心和仓位建议
    """
    today = datetime.today().strftime("%Y-%m-%d")
    signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}

    records = [
        {
            "Date": today,
            "Ticker": ticker,
            "Signal": signal_map.get(signal_info[0], "UNKNOWN"),
            "Confidence": f"{signal_info[1]*100:.1f}%",
            "Position Size": f"{int(signal_info[2]*100)}%"
        }
        for ticker, signal_info in signals_dict.items()
    ]

    record_df = pd.DataFrame(records)

    os.makedirs("trading_records", exist_ok=True)
    record_file = "trading_records/trade_signals_history.csv"

    if not os.path.exists(record_file):
        record_df.to_csv(record_file, index=False)
    else:
        record_df.to_csv(record_file, mode='a', header=False, index=False)

    print(f"[INFO] Today's detailed signals saved to {record_file}")

def main():
    tickers = [
        "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "NVDA",
        "JPM", "BAC", "WFC", "V", "MA", "UNH", "PFE", "MRK",
        "XOM", "CVX", "COP", "HD", "COST", "DIS", "NFLX", "MCD",
        "KO", "PEP", "NKE", "ORCL", "INTC", "CSCO", "ABT"
    ]
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. 下载原始数据
    download_stock_data(tickers, start_date, end_date)

    # 2. 加载并提取特征
    data_dict = load_and_prepare_data(tickers, start_date, end_date)

    # 3. 训练交易信号模型
    train_signal_model(data_dict)

    # 4. 预测今天的交易信号
    signals = predict_signals("models/trading_model.pkl", data_dict)

    # 5. 打印今天交易信号 + 仓位建议
    signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
    result = pd.DataFrame([
        {
            "Ticker": ticker,
            "Signal": signal_map.get(signal[0], "UNKNOWN"),
            "Confidence": f"{signal[1]*100:.1f}%",
            "Position Size": f"{int(signal[2]*100)}%"
        }
        for ticker, signal in signals.items()
    ])
    print("\n=== Today's Trading Signals with Position Suggestion ===")
    print(result.to_string(index=False))


    # 6. 保存今天交易记录
    save_trading_signals(signals)

if __name__ == "__main__":
    main()
