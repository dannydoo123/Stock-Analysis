import pandas as pd
import numpy as np

def prepare_training_data(data):
    """
    Prepare training data for the Random Forest model.
    - Shifts the 'close' price forward by 1 day for future price movement
    - X: RSI, MACD, MACD_signal, SMA_50, OBV
    - y: 1 if next day's close is higher than today's, otherwise 0
    """

    # 1) Shift 'close' to define 'Future_Close'
    data["Future_Close"] = data["close"].shift(-1)

    # 2) Define Target = 1 if future price is higher, else 0
    data["Target"] = (data["Future_Close"] > data["close"]).astype(int)

    # 3) Combine indicator columns + 'Target' in one DataFrame
    #    Make sure these columns exist in 'data' before referencing them
    combined = data[[
        "RSI_14", "MACD", "MACD_signal", 
        "SMA_50", "OBV", 
        "Target"  # Our classification label
    ]].copy()

    # 4) Drop rows with any NaN to ensure alignment
    combined.dropna(inplace=True)

    # 5) Separate features (X) and labels (y)
    X = combined[["RSI_14", "MACD", "MACD_signal", "SMA_50", "OBV"]]
    y = combined["Target"]

    return X, y
