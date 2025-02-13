import pandas as pd

def generate_trading_signals(data):
    """
    Generate refined Buy/Sell/Hold signals based on RSI, MACD, SMA, EMA, and OBV (Swing Trading Focus).
    """

    signals = pd.DataFrame(index=data.index)

    # RSI-Based Trading Logic (Momentum Indicator)
    signals["RSI_Signal"] = "HOLD"
    signals.loc[data["RSI_14"] < 30, "RSI_Signal"] = "BUY"  # Oversold condition
    signals.loc[data["RSI_14"] > 70, "RSI_Signal"] = "SELL"  # Overbought condition

    # MACD-Based Trading Logic (Trend & Momentum)
    signals["MACD_Signal"] = "HOLD"
    signals.loc[data["MACD"] > data["MACD_signal"], "MACD_Signal"] = "BUY"  # Bullish crossover
    signals.loc[data["MACD"] < data["MACD_signal"], "MACD_Signal"] = "SELL"  # Bearish crossover

    # SMA Trend Confirmation (Trend Following)
    signals["Trend_Signal"] = "HOLD"
    signals.loc[data["close"] > data["SMA_50"], "Trend_Signal"] = "BUY"  # Uptrend confirmation
    signals.loc[data["close"] < data["SMA_50"], "Trend_Signal"] = "SELL"  # Downtrend confirmation

    # OBV-Based Confirmation (Volume Trend Analysis)
    signals["OBV_Signal"] = "HOLD"
    signals.loc[data["OBV"] > data["OBV"].rolling(10).mean(), "OBV_Signal"] = "BUY"  # Rising volume trend
    signals.loc[data["OBV"] < data["OBV"].rolling(10).mean(), "OBV_Signal"] = "SELL"  # Falling volume trend

    # # Commented out signals for potential future use:
    # signals["ATR_Signal"] = "HOLD"
    # signals.loc[data["ATR_14"] > data["ATR_14"].rolling(10).mean(), "ATR_Signal"] = "HIGH_VOLATILITY"

    # signals["Stoch_Signal"] = "HOLD"
    # signals.loc[data["Stoch_K"] < 20, "Stoch_Signal"] = "BUY"
    # signals.loc[data["Stoch_K"] > 80, "Stoch_Signal"] = "SELL"

    # signals["BB_Signal"] = "HOLD"
    # signals.loc[data["close"] < data["BB_lower"], "BB_Signal"] = "BUY"
    # signals.loc[data["close"] > data["BB_upper"], "BB_Signal"] = "SELL"

    # Weighted Scoring System for Final Signal
    signal_weights = {
        "RSI_Signal": 1,   # RSI contributes 1 point
        "MACD_Signal": 2,  # MACD is weighted higher for stronger trend confirmation
        "Trend_Signal": 1,  # SMA 50 as trend filter
        "OBV_Signal": 1    # OBV volume confirmation
    }

    # Explicitly Convert Categorical Signals into Numerical Format Using `.map()`
    # `map()` ensures explicit conversion of categories to numerical values
    # `fillna(0)` prevents missing values from causing errors
    signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}

    score = (
        signals["RSI_Signal"].map(signal_map).fillna(0).astype(int) * signal_weights["RSI_Signal"] +
        signals["MACD_Signal"].map(signal_map).fillna(0).astype(int) * signal_weights["MACD_Signal"] +
        signals["Trend_Signal"].map(signal_map).fillna(0).astype(int) * signal_weights["Trend_Signal"] +
        signals["OBV_Signal"].map(signal_map).fillna(0).astype(int) * signal_weights["OBV_Signal"]
    )

    # Final Trading Signal Based on Weighted Score
    signals["Final_Signal"] = "HOLD"
    signals.loc[score >= 2, "Final_Signal"] = "BUY"  # Strong buy signal if score is high
    signals.loc[score <= -2, "Final_Signal"] = "SELL"  # Strong sell signal if score is low

    return signals
