import pandas as pd, numpy as np
import yfinance as yf
import requests, argparse, os
from io import StringIO
import pandas as pd
import warnings
from .common import SP500_NAMES_FILE, START_STORE_DATE, DATA_INTERVAL,SP500_LIST_URL, DEBUG_SYMBOLS, get_market_file
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
def test_subset(df, args):
    if args.debug:
        print(f"returning the following symbols :{DEBUG_SYMBOLS}")
        return(df[df['Symbol'].isin(DEBUG_SYMBOLS)])
    elif args.test: 
        print("returning only test subset")
        return df.sort_values('Date added').head(50)
    else:
        return df
def fetch_sp500_list(filepath, args, url=SP500_LIST_URL, headers=headers):
    '''Download SP500 list of companies'''
    if args.newtable or os.path.getsize(filepath)<1:
        print("Downloading the sp500 list")
        response = requests.get(SP500_LIST_URL, headers=headers)
        response.raise_for_status()  # Ensure we got a 200 OK

        # Parse HTML content with pandas
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        df.to_csv(filepath)
        print(f"Saved to {filepath}")

    else:   
        print(f"File {filepath} should be already available")
        df = pd.read_csv(filepath)
    df = test_subset(df,args)
    return df

def fetch_sp500_marketdata(filepath, tickers, force_download=False, monthly="end"):
    '''Use Yfinance api to download data for sp500 companies (and the SPYear return)'''
    if force_download or os.path.exists(filepath) is False:
        px = yf.download(tickers, start=START_STORE_DATE,
                         interval=DATA_INTERVAL, auto_adjust=True,
                         progress=False)["Close"]
        spy = yf.download("SPY", start=START_STORE_DATE,
                          interval=DATA_INTERVAL, auto_adjust=True,
                          progress=False)["Close"]

        if monthly == "end":
            freq = "BME"  # business month-end
            px_m = px.resample(freq).last()
            spy_m = spy.resample(freq).last()
        elif monthly == "start":
            freq = "BMS"  # business month-start
            px_m = px.resample(freq).first()
            spy_m = spy.resample(freq).first()
        else:
            raise ValueError("monthly must be 'end' or 'start'")

        # Save wide monthly prices; write SPY 12M returns alongside if desired
        out = px_m.copy()
        out.to_csv(filepath)
        print(f"Saved to {filepath}")
        return px_m, spy_m
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
    sp500_marketfile = get_market_file(args)
    all_tickers = add_SPY(tickers)
    fetch_sp500_marketdata(sp500_marketfile, all_tickers, args.newinfo)


