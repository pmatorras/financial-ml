import requests, json, re, csv, os, pickle
import pandas as pd
import requests, sys

from bs4          import BeautifulSoup
from datetime     import datetime
from googlesearch import search

now    = datetime.now()
today  = datetime.date(now)
foldir = os.path.dirname(sys.argv[0])
if "/" in foldir: foldir += "/"


CRED = '\033[91m'
CEND = '\033[0m'

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'}
stocks_esp = {"LOGISTA"             : ["logista_hlgd_sa"       ,"56747"],
              "ORYZON GENOMICS"     : ["oryzon_genomics_sa"    ,"57000"] ,
              "GRENERGY RENOVABLES" : ["grenergy_renovables_sa","56988"]}

linkbase   = {"cnn" : "https://money.cnn.com/quote/forecast/forecast.html?symb=",
              "wsj" : "https://www.wsj.com/market-data/quotes/",
              "esp" : "https://cincodias.elpais.com/mercados/empresas/"}


def makeSoup(link):
    print link
    request   = requests.get(link, headers=headers)
    return  BeautifulSoup(request.text,"lxml")



jsonDict  = open("full_portfolio.json", "rb")
portfolio = json.load(jsonDict)
jsonDict.close()
stocks_csv={}
def loopqueries(query,recom,links):
    for j in search(query, tld='com', num =2, stop=2, pause =2):
        if recom.lstrip() not in j: continue 
        links.append(j)

def getLinksGoogle(site, stock, recom):
    links = []
    query = site+ stock["symbol"]+" "+stock["name"].replace("SA","")+" "+recom
    print "Stock", stock["symbol"], "\t query", query

    loopqueries(query,recom,links)

    if len(links) == 0 :
        query = site + stock["symbol"].replace("SA","")+ recom 
        loopqueries(query,recom, links)
    if len(links) == 0 :
        query = site + stock["name"].replace("SA","")+ recom 
        loopqueries(query,recom, links)
    print "links", links[0]
    return links
    

for entry in portfolio:
    if "IE" in portfolio[entry]["isin"]:
        print "ETF", entry
        continue

    stocks_csv[portfolio[entry]["symbol"]] =  [portfolio[entry]["name"], portfolio[entry]["isin"]]
    
    if "ES" in portfolio[entry]["isin"]:
        links = getLinksGoogle("site:cincodias.elpais.com/mercados/empresas/ ", portfolio[entry], " recomendaciones")
        stocks_csv[portfolio[entry]["symbol"]].append(links[0])
    else: # "US" in portfolio[entry]["isin"]:
        links = getLinksGoogle("site:https://www.wsj.com/market-data/quotes/ ", portfolio[entry], " research-ratings")
        stocks_csv[portfolio[entry]["symbol"]].append(links[0])
        



f = open(foldir+"Portfolio_dict.pkl","wb")
pickle.dump(stocks_csv,f)
f.close()
