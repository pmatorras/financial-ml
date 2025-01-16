import requests, json, re, csv, os, pickle, sys, optparse
import pandas as pd
import numpy  as np
from bs4                import BeautifulSoup
from datetime           import datetime
from currency_converter import CurrencyConverter


# Headers definition remains the same
headers = {
    "1": {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'},
    "2": {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'}
}

# Get the current working directory and script directory
srcdir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Define directories using os.path.join for cross-platform compatibility
datadir = os.path.join(srcdir, "..", "Data")
plotdir = os.path.join(srcdir, "..", "Plots")
foredir = os.path.join(plotdir, "Forecast")

# File paths
full_port = os.path.join(datadir, "full_portfolio.json")
port_pkl = os.path.join(datadir, "Portfolio_dict.pkl")
act_info = os.path.join(datadir, "act_info.json")

print("Data directory:", datadir)

# Create directories if they don't exist
os.makedirs(datadir, exist_ok=True)
os.makedirs(plotdir, exist_ok=True)
os.makedirs(foredir, exist_ok=True)




