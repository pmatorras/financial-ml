import requests, json, re, csv, os, pickle, requests, sys, optparse
import pandas as pd
import numpy  as np
from bs4                import BeautifulSoup
from datetime           import datetime
from currency_converter import CurrencyConverter

headers = {"1": {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'},
           "2": {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'}}
srcdir  = os.getcwd()+'/'+os.path.dirname(sys.argv[0])
if "/" not in srcdir[-1]: srcdir += "/"
datadir   = srcdir  + "../Data/"
plotdir   = srcdir  + "../Plots/"
foredir   = plotdir + "Forecast/"
full_port = datadir + "full_portfolio.json" 
port_pkl  = datadir + "Portfolio_dict.pkl"
act_info  = datadir + "act_info.json" 

os.system('mkdir -p ' + datadir)
os.system('mkdir -p ' + plotdir)
os.system('mkdir -p ' + foredir)



