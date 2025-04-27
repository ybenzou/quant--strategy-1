import os
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
