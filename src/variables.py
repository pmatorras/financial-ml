import requests, json, re, csv, os, pickle, requests, sys, optparse
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'}
srcdir  = os.getcwd()+'/'+os.path.dirname(sys.argv[0])
if "/" not in srcdir[-1]: srcdir += "/"
datadir = srcdir + "../Data/"
os.system('mkdir -p ' + datadir)
