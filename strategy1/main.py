from data_fetcher.fetcher import fetch_stock_data
from utils.visualization import plot_stock_price_and_volume

def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    data = fetch_stock_data(ticker, start_date, end_date)
    plot_stock_price_and_volume(data, ticker)

if __name__ == "__main__":
    main()
