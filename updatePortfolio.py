import requests, json, re, csv, os, pickle
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import requests, sys
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

#Find symbol similar to name
def get_symbol(stock, country):
    url = "https://www.marketwatch.com/tools/quotes/lookup.asp?siteID=mktw&Lookup="+stock+"&Country="+country+"&Type=all"#Stock"
    print stock, country
    print url
    soup    = makeSoup(url)
    results = soup.findAll(class_="results")
    if stock[:3] not in str(results[0]).split("title=")[1].split(">")[0].upper():
        print CRED + "Stock: " + stock + " different from " + str(results[0]).split("title=")[1].split(">")[0] + CEND
        symbol = 'NOT_DO'
    else:
        symbol = str(results[0]).split("title=")[1].split(">")[1].split("<")[0]
        if "VOW" in symbol: symbol += "3" 
        print "symbol:","\033[34m"+ symbol + CEND
    return symbol
#Read names from Portfolio.csv
def readNames(inputfile):
    print inputfile
    with open(inputfile) as csvfile:
        reader     = csv.reader(csvfile, delimiter=',')
        names      = []
        countries  = []
        currencies = []
        symbols    = []
        for row in reader:
            if "cash" in row[0].lower() or "Producto" in row[0] or "ISHARES" in row[0] : continue
            long_name = row[0].split(' ')
            name      = row[0]
            country   = row[1][:2]
            currency  = row[4][:3]
            if "NL" in country and "prosus" not in name.lower():
                if "USD" in currency: country = "US"
                else: country = "DE"
            if long_name[len(long_name)-1]  == "COMM": name = name.replace('COMM','')
            name = name.replace("REGISTERED","").replace("NY","").replace("S.A.","").replace("INC","").replace(" ","+").replace("GROUP","").replace("HOLDING","").strip()
            if "TOTAL"  in name: name +="Energies"
            names     .append(name)
            countries .append(country)
            currencies.append(currency)
            symbols   .append(get_symbol(name, country))
    return [names,symbols, currencies, countries]


from googlesearch import search


jsonDict  = open("full_portfolio.json", "rb")
portfolio = json.load(jsonDict)
jsonDict.close()
stocks_csv={}

def getLinksGoogle(site, stock, reconm):
    links = []
    query = "site:cincodias.elpais.com/mercados/empresas/ "+ stock["name"]+ reconm
    for j in search(query, tld='com', num =2, stop=2, pause =2):
        if "recomendaciones" not in j: continue 
        links.append(j)
    if len(links) == 0 :
        query = site+ stock["symbol"]+ reconm
        for j in search(query, tld='com', num =2, stop=2, pause =2):
            if "recomendaciones" not in j: continue 
            links.append(j)
    print "returning", links
    return links

for entry in portfolio:
    stocks_csv[portfolio[entry]["symbol"]] =  [portfolio[entry]["name"], portfolio[entry]["isin"]]
    if "ES" in portfolio[entry]["isin"]:
        print "query", query
        links = getLinksGoogle("site:cincodias.elpais.com/mercados/empresas/ ", portfolio[entry], " recomendaciones")
        stocks_csv[portfolio[entry]["symbol"]].append(links[0])
    elif "US" in portfolio[entry]["isin"]:
        query = "site:cincodias.elpais.com/mercados/empresas/ "+ portfolio[entry]["name"]+ " recomendaciones"

    

print stocks_csv






#old way
''' 
print foldir
results    = readNames(foldir+"Portfolio.csv")

stocks_csv = {}

for idx, result in enumerate(results[0]):
    result_i = result.replace("+"," ")
    if result_i in stocks_esp.keys(): stocks_csv[result_i] = [stocks_esp[result_i][0], stocks_esp[result_i][1],"ESP"]
    else: stocks_csv[result_i] = [results[1][idx], results[2][idx], results[3][idx]]
'''

f = open(foldir+"Portfolio_dict.pkl","wb")
pickle.dump(stocks_csv,f)
f.close()
