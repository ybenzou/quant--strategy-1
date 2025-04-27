# Moving average strategy placeholder

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
