import pandas as pd
import requests
from config import ALPHA_VANTAGE_API_KEY

def get_stock_data(symbol):
    """
    Fetches FULL historical stock data from AlphaVantage.
    """
    print(f"🔄 Requesting full historical data for {symbol}...")

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",  # Changed from "compact" to "full"
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"❌ Error: Could not retrieve data for {symbol}. Check API key or symbol.")
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })

    # Convert date index to datetime and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"✅ Successfully retrieved {len(df)} days of data for {symbol}.")
    return df
