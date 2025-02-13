import os
import pandas as pd

# Data fetching & indicators
from data.fetch_data import get_stock_data
from indicators.technical import calculate_indicators

# Rule-based signals
from insights.analysis import generate_trading_signals

# AI model training (Random Forest)
from model.train_models import train_models

def main():
    """
    AI Trading Program:
    1) Fetch stock data & calculate indicators.
    2) (Optional) Display the last N days of trading insights (RSI, MACD, SMA, OBV, etc.).
    3) Train a Random Forest model on the data.
    """

    # Get user input for tickers
    tickers = input("Enter stock symbols (comma-separated): ").strip().upper().split(",")

    # Ask if the user wants to see trading insights
    show_insights = input("Do you want to see trading insights? (y/n): ").lower().startswith("y")

    # If showing insights, ask how many days to display
    num_days = 5  # Default
    if show_insights:
        while True:
            try:
                num_days = int(input("How many days of trading insights would you like to see? "))
                if num_days > 0:
                    break
                else:
                    print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

    # Process each ticker
    for ticker in tickers:
        ticker = ticker.strip()
        print(f"\n🔄 Requesting data for {ticker}...")

        # Step 1: Fetch data
        data = get_stock_data(ticker)

        if data is not None:
            # Step 2: Calculate technical indicators
            data = calculate_indicators(data)

            # Step 3: (Optional) Generate and display rule-based insights
            if show_insights:
                signals = generate_trading_signals(data)
                data = data.join(signals)

                print(f"\n=== Trading Insights for {ticker} (Last {num_days} Days) ===")
                # Ensure required columns exist
                columns_to_show = [
                    "RSI_14", "MACD", "MACD_signal",
                    "SMA_50", "OBV",  # Key indicators
                    "RSI_Signal", "MACD_Signal", "Trend_Signal", "OBV_Signal", "Final_Signal"
                ]
                available_cols = [col for col in columns_to_show if col in data.columns]
                print(data[available_cols].tail(num_days))

            # Step 4: Train Random Forest model on the data
            print(f"\n📊 Training AI model for {ticker}...")
            train_models(data)
        else:
            print(f"⚠️ Failed to retrieve data for {ticker}")

if __name__ == "__main__":
    main()
