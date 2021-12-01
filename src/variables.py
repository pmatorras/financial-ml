import requests, json, re, csv, os, pickle, requests, sys, optparse
import pandas as pd
from bs4      import BeautifulSoup
from datetime import datetime
CRED    = '\033[91m'
CEND    = '\033[0m'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'}
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


