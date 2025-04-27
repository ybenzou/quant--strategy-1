# Backtester placeholder

class Backtester:
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals

    def run_backtest(self):
        returns = self.data["Close"].pct_change()
        strategy_returns = returns * self.signals.shift(1)
        cumulative_returns = (1 + strategy_returns).cumprod()
        return cumulative_returns
