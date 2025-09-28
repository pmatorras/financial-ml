import pandas as pd, numpy as np
import yfinance as yf
import requests, argparse, os
from io import StringIO
import pandas as pd
import warnings
from .common import SP500_MARKET_TEST,SP500_MARKET_FILE,SP500_NAMES_FILE
sp500_list_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
def fetch_sp500_list(filepath, url, headers, force_download=False):
    if force_download or os.path.getsize(filepath)<1:
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
    return df

def fetch_sp500_marketdata(filepath, tickers, force_download=False):
    if force_download or os.path.exists(filepath) is False:
        print("Downloading the sp500 list")
        date_start="1995-01-01"
        sp500_data = yf.download(tickers, start=date_start, interval="1mo", auto_adjust=True, progress=False)["Close"]
        spy_data = yf.download("SPY", start=date_start, interval="1mo", auto_adjust=True, progress=False)["Close"]
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

def store_info(args):
    spx = fetch_sp500_list(SP500_NAMES_FILE, sp500_list_url, headers, args.newtable)
    all_tickers = spx["Symbol"].str.replace(".", "-", regex=False).tolist()  

    if args.test:
        oldest_stocks = spx.sort_values('Date added').head(50)
        subset_tickers = oldest_stocks['Symbol'].tolist()
        add_SPY(subset_tickers)
        fetch_sp500_marketdata(SP500_MARKET_TEST, subset_tickers, args.newinfo)
    else:
        add_SPY(all_tickers)
        fetch_sp500_marketdata(SP500_MARKET_FILE, all_tickers, args.newinfo)

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GDP and Inflation for selected countries")
    parser.add_argument("-nt", "--newtable", action="store_true", help="Update sp500 table")    
    parser.add_argument("-ni", "--newinfo", action="store_true", help="Update sp500 financial information") 
    parser.add_argument("-t" , "--test"   , action="store_true", help="Test on a smaller subset of 50")    
   
    args = parser.parse_args()
    main()
'''