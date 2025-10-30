"""
Collect sentiment indicators (VIX, put/call ratio, etc.)
"""
import pandas as pd
import yfinance as yf
import os
from financial_ml.utils.config import START_STORE_DATE, DATA_INTERVAL
from financial_ml.utils.paths import get_sentiment_file




def fetch_vix_data(filepath, force_download=True, monthly="end"):
    """
    Download VIX data from Yahoo Finance and save to CSV
    
    Args:
        filepath: Path to save VIX CSV
        force_download: Force re-download even if file exists
        monthly: 'end' or 'start' for month-end/start resampling
    
    Returns:
        pd.Series with monthly VIX close prices
    """
    if force_download or not os.path.exists(filepath):
        print("Downloading VIX data from Yahoo Finance...")
        
        # Download VIX (CBOE Volatility Index)
        print(START_STORE_DATE, DATA_INTERVAL)
        vix = yf.download('^VIX', start=START_STORE_DATE,
                         interval=DATA_INTERVAL, auto_adjust=True,
                         progress=False)['Close']
        
        # Resample to monthly (matching your market data pattern)
        if monthly == "end":
            freq = "BME"  # business month-end
            vix_m = vix.resample(freq).last()
        elif monthly == "start":
            freq = "BMS"  # business month-start
            vix_m = vix.resample(freq).first()
        else:
            raise ValueError("monthly must be 'end' or 'start'")
        
        # Save to CSV (single column format)
        vix_m.to_csv(filepath, header=['VIX'])
        print(f"Saved VIX data to {filepath}")
        return vix_m
    else:
        print(f"File {filepath} already exists")
        return None


def collect_sentiment_data(args):
    """
    Main function to collect sentiment indicators
    
    Args:
        args: Namespace with newinfo flag for force download
    """
    sentiment_file = get_sentiment_file(args)
    
    # Download VIX (more indicators can be added here later)
    fetch_vix_data(sentiment_file)
    
    print("Sentiment data collection complete")
