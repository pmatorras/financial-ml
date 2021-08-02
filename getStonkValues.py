import requests, json, re, csv, os, pickle, requests, sys
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

#Define basic variables
now     = datetime.now()
today   = datetime.date(now)
foldir  = os.path.dirname(sys.argv[0])
CRED    = '\033[91m'
CEND    = '\033[0m'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'}
csvname = 'stocks.txt'
if '/' in foldir: foldir+='/'
#Stock dictionaries
stocks_wsj = {"PROSUS N.V." : ["PRX" , "NL/XAMS/"],
              "Volkswagen"  : ["VOW3", "XE/XETR/"]}
stocks_esp = {"LOGISTA"             : ["logista_hlgd_sa"       ,"56747"],
              "ORYZON GENOMICS"     : ["oryzon_genomics_sa"    ,"57000"],
              "GRENERGY RENOVABLES" : ["grenergy_renovables_sa","56988"]}

stocks_cnn = {"Alibaba" : ["BABA"], "Airbus" : ["EADSY"], "Curevac"   : ["CVAC"],
              "Arcelor" : ["MT"  ], "TSMC  " : ["TSM"  ], "Dr Horton" : ["DHI" ],
              "Total"   : ["TOT"]}

linkbase   = {"cnn" : "https://money.cnn.com/quote/forecast/forecast.html?symb=",
              "wsj" : "https://www.wsj.com/market-data/quotes/",
              "esp" : "https://cincodias.elpais.com/mercados/empresas/"}

def getBetween(string, before, after):
    return string.split(before)[1].split(after)[0]

def writeCSV(csvname,stock, stocksym, act_val, exp_val, exp_max, exp_min,recos, nrecos):
    if "/" in csvname: csvname =  foldir+"/"+csvname
    csvexist= os.path.exists(csvname)
    if csvexist is False: w_type = 'w'
    else: w_type = 'a'
    with open(csvname, w_type) as f:
        writer   = csv.writer(f, delimiter = '\t')
        titlerow = stock+"\t["+stocksym+"]\t" 
        if csvexist is False : writer.writerow(['stock','abrv' , 'variable', str(today)])
        writer.writerow([stock, stocksym, "act_val", act_val  ])
        writer.writerow([stock, stocksym, "exp_val", exp_val  ])
        writer.writerow([stock, stocksym, "exp_max", exp_max  ])
        writer.writerow([stock, stocksym, "exp_min", exp_min  ])
        writer.writerow([stock, stocksym, "buy"    , recos[0] ])
        writer.writerow([stock, stocksym, "overw"  , recos[1] ])
        writer.writerow([stock, stocksym, "hold"   , recos[2] ])
        writer.writerow([stock, stocksym, "underw" , recos[3] ])
        writer.writerow([stock, stocksym, "sell"   , recos[4] ])
        writer.writerow([stock, stocksym, "n_recom", nrecos   ])
        writer.writerow("")



def printValues(stock, stocksym, act_val, exp_val, exp_max, exp_min,  exp_perc, nrecos, nmonths):
    col_ini = CEND
    if   "-" in str(exp_perc) or exp_perc<0 : col_ini = '\033[31m'
    elif "+" in str(exp_perc) or exp_perc>0 : col_ini =  '\033[32m'
    if float(exp_perc)>0: exp_perc = "+"+str(float(exp_perc))
    col_end = CEND
    print stock, "["+stocksym+"]"
    print "current value:", '\033[34m'+act_val+col_end
    print "expected value in" ,nmonths, "months:", exp_val, "all", nrecos, "analysis fall within ["+exp_max+","+exp_min+"]"
    print "Current average gain:", col_ini+str(exp_perc)+col_end
    print ""

def makeSoup(link):
    print link
    request   = requests.get(link, headers=headers)
    return  BeautifulSoup(request.text,"lxml")


def getStocks(stocks, type_i):
    for stocksym in stocks.keys():
        if type_i.lower() in ["esp", "cnn", "wsj"]:  stockType   = type_i
        else:
            if "https" in stocks[stocksym][-1].lower():
                link = stocks[stocksym][-1]
                if "cincodias" in link: stockType = "esp"
                elif     "cnn" in link: stockType = "cnn"
                elif     "wsj" in link: stockType = "wsj"
                else:
                    print "unknown webpage"
                    continue
            elif "esp" in stocks[stocksym][-1].lower():
                stockType = 'esp'
            elif ":"   in stocks[stocksym][0]:
                stockType ='wsj'
            else:
                stockType = 'cnn'
        nmonths = '12'
        
        stock = stocks[stocksym][0]
        print stocksym, stocks[stocksym], stock
        soup       = makeSoup(link)

        if "wsj" in stockType:
            reco_table =  soup.find(class_="cr_analystRatings cr_data module").find(class_="cr_dataTable").findAll(class_="data_data")
            recos      = []
            for i in range(2,len(reco_table),3):
                recos.append(int(getBetween(str(reco_table[i]),">","<")))

            nrecos = sum(recos)
            prices = soup.find(class_="cr_data rr_stockprice module").findAll(class_="data_data")

            exp_max  = getBetween(str(prices[0]), "/sup>", "</span")
            exp_min  = getBetween(str(prices[2]), "/sup>", "</span")
            exp_med  = getBetween(str(prices[1]), "/sup>", "</span")
            exp_avg  = getBetween(str(prices[3]), "/sup>", "</span")
            act_val  = getBetween(str(prices[4]), "/sup>", "</span")
            exp_perc = round(100*(float(exp_med)-float(act_val))/float(act_val),2)

        elif "esp" in stockType:
            dataset     = soup.text.split("var barChartData =")[1].split('};')[0]
            recommend   =  soup.text.split("Tendencia de las recomendaciones")[1].split("*La")[0]
            all_act_val = getBetween(dataset.split('Precio real')[1].split('data')[1], '[', ']')
            all_exp_med = getBetween(dataset.split('Precio objetivo')[1].split('data')[1], '[', ']')
            act_val     = all_act_val.replace("\n\n","").split(',')[1]
            exp_med     = all_exp_med.replace("\n\n","").split(',')[1]
            exp_max     = exp_med
            exp_min     = exp_med
            recos       = recommend.split("Hoy\n")[1].split("*")[0].split("\n")
            nrecos      = recos[5]
            exp_perc    = round(100*(float(exp_med)-float(act_val))/float(act_val),2)

        elif "cnn" in stockType:
            valheader = soup.find(class_='wsod_last')
            act_val   = getBetween(str(valheader), '"ToHundredth">', "</span")
            name      = str(soup.find(class_="wsod_fLeft wsod_narrowH1Container"))
            forecast  = str(soup.find(class_='wsod_twoCol clearfix'))
            numbers   =  re.findall(r"[-+]?\d*\.\d+|\d+", forecast)
            nrecos    = numbers[0]
            nmonths   = numbers[1]
            exp_med   = numbers[2]
            exp_max   = numbers[3]
            exp_min   = numbers[4]
            exp_perc  = numbers[5]
            act_val2  = numbers[6]

            col_ini = CEND
            if   "-" in exp_perc : col_ini = '\033[31m'
            elif "+" in exp_perc : col_ini =  '\033[32m'
            col_end = CEND

            if float(act_val) != float(act_val2):
                print CRED + "Error, numbers "+act_val+" and "+act_val2+" are different!" + CEND
            recos = [0,0,0,0,0]

        else:
            print "unrecognised stock Type", stockType
            continue
        print "STOCK", stock, stocksym
        printValues(stock, stocksym, act_val, exp_med, exp_max, exp_min, exp_perc, nrecos, nmonths)

        writeCSV(csvname, stock, stocksym, act_val, exp_med, exp_max, exp_min, recos, nrecos)
        


   
pickleDict = open(foldir+"Portfolio_dict.pkl", "rb")
portfolio  = pickle.load(pickleDict)
pickleDict.close()
os.system('rm '+csvname)

getStocks(portfolio, "multiple")
