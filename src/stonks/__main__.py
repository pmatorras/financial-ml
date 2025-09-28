import pandas as pd, numpy as np
import yfinance as yf
import requests, argparse, os
from io import StringIO
import pandas as pd
from .store_info import store_info
from .train import train
def main():
    parser = argparse.ArgumentParser(description="Compare GDP and Inflation for selected countries")
    parser.add_argument("-nt", "--newtable", action="store_true", help="Update sp500 table")    
    parser.add_argument("-ni", "--newinfo", action="store_true", help="Update sp500 financial information") 
    parser.add_argument("--test"   , action="store_true", help="Test on a smaller subset of 50")    
    parser.add_argument("--notrain"   , action="store_true", help="Test on a smaller subset of 50")   
    args = parser.parse_args()
    store_info(args)
    if args.notrain is None: train(args)



if __name__ == "__main__":
    main()