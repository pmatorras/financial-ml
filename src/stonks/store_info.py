import pandas as pd, numpy as np
import yfinance as yf
import requests, argparse, os
from io import StringIO
import pandas as pd
import common
sp500_list_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
def fetch_sp500_list(filepath, url, headers, force_download=False):
    if force_download or os.path.exists(filepath) is False:
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
    #print(spx.head())
def main():
    spx = fetch_sp500_list(common.SP500NM_FILE, sp500_list_url, headers, args.newtable)
    tickers = spx["Symbol"].str.replace(".", "-", regex=False).tolist()  # adjust BRK.B -> BRK-B, etc.
    oldest_stocks = spx.sort_values('Date added').head(50)
    # Get tickers for the oldest 50 firms
    subset_tickers = oldest_stocks['Symbol'].tolist()
    print(subset_tickers)
    px = fetch_sp500_marketdata(common.SP500MARKET_FILE, subset_tickers, args.newinfo)
def fetch_sp500_marketdata(filepath, tickers, force_download=False):
    if force_download or os.path.exists(filepath) is False:
        print("Downloading the sp500 list")
        sp500_data = yf.download(tickers, start="2005-01-01", interval="1mo", auto_adjust=True, progress=False)["Close"]
        print("sp500_data", sp500_data)
        sp500_data.to_csv(filepath)
        print(f"Saved to {filepath}")
    else: 
        print(f"File {filepath} should be already available")
        sp500_data = pd.read_csv(filepath)
    return sp500_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GDP and Inflation for selected countries")
    parser.add_argument("-nt", "--newtable", action="store_true", help="Update sp500 table")    
    parser.add_argument("-ni", "--newinfo", action="store_true", help="Update sp500 financial information")    
    args = parser.parse_args()
    main()