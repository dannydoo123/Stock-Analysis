import talib
import pandas as pd

def calculate_rsi(data, window=14):
    return talib.RSI(data["close"], timeperiod=window)

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    macd, signal, _ = talib.MACD(data["close"], fastperiod=short_window, slowperiod=long_window, signalperiod=signal_window)
    return macd, signal

def calculate_sma(data, window=50):
    return talib.SMA(data["close"], timeperiod=window)

def calculate_ema(data, window=50):
    return talib.EMA(data["close"], timeperiod=window)

def calculate_obv(data):
    return talib.OBV(data["close"], data["volume"])

def calculate_indicators(data):
    """
    Compute multiple technical indicators using modular functions.
    """

    # ✅ Use individual functions for better readability
    data["RSI_14"] = calculate_rsi(data)
    data["MACD"], data["MACD_signal"] = calculate_macd(data)
    data["SMA_50"] = calculate_sma(data, window=50)
    data["EMA_20"] = calculate_ema(data, window=20)
    data["OBV"] = calculate_obv(data)

    # ✅ Additional indicators
    data["SMA_200"] = calculate_sma(data, window=200)
    data["ATR_14"] = talib.ATR(data["high"], data["low"], data["close"], timeperiod=14)
    data["STOCH_K"], data["STOCH_D"] = talib.STOCH(data["high"], data["low"], data["close"])
    data["ADX"] = talib.ADX(data["high"], data["low"], data["close"], timeperiod=14)
    data["CCI"] = talib.CCI(data["high"], data["low"], data["close"], timeperiod=14)
    data["Williams_R"] = talib.WILLR(data["high"], data["low"], data["close"], timeperiod=14)

    return data
