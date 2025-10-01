import pandas as pd, numpy as np
import yfinance as yf
import requests, argparse, os
from io import StringIO
import pandas as pd
import warnings
from .common import SP500_MARKET_TEST,SP500_MARKET_FILE,SP500_NAMES_FILE, START_STORE_DATE, DATA_INTERVAL,SP500_LIST_URL
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}

def fetch_sp500_list(filepath, args, url=None, headers=None):
    '''Download SP500 list of companies'''
    if args.newtable or os.path.getsize(filepath)<1:
        print("Downloading the sp500 list")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Ensure we got a 200 OK

        # Parse HTML content with pandas
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        df.to_csv(filepath)
        print(f"Saved to {filepath}")

    else:   
        print(f"File {filepath} should be already available")
        df = pd.read_csv(filepath)
    if args.test: 
        print("returning only test subset")
        df=df.sort_values('Date added').head(50)
    return df

def fetch_sp500_marketdata(filepath, tickers, force_download=False):
    '''Use Yfinance api to download data for sp500 companies (and the SPYear return)'''
    if force_download or os.path.exists(filepath) is False:
        print("Downloading the sp500 list")
        sp500_data = yf.download(tickers, start=START_STORE_DATE, interval=DATA_INTERVAL, auto_adjust=True, progress=False)["Close"]
        spy_data = yf.download("SPY", start=START_STORE_DATE, interval=DATA_INTERVAL, auto_adjust=True, progress=False)["Close"]
        print("sp500_data", sp500_data, "SPY data", spy_data)
        sp500_data.to_csv(filepath)
        print(f"Saved to {filepath}")
    else: 
        print(f"File {filepath} should be already available")

def add_SPY(tickers):    
    if "SPY" in tickers:
        warnings.warn(f"SPY is found in list!", UserWarning)
    else:
        tickers.append("SPY")
    return tickers
def store_info(args):
    spx = fetch_sp500_list(SP500_NAMES_FILE, args, SP500_LIST_URL, headers)
    tickers = spx["Symbol"].str.replace(".", "-", regex=False).tolist()  
    sp500_marketfile = SP500_MARKET_TEST if args.test else SP500_MARKET_FILE
    all_tickers = add_SPY(tickers)
    fetch_sp500_marketdata(sp500_marketfile, all_tickers, args.newinfo)


