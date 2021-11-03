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
if '/' in foldir: foldir+='/'
jsonFile   = open("full_portfolio.json", "rb")
jsonStocks = json.load(jsonFile)
symb_isin  = {}
for entry in jsonStocks:
    symb_isin[jsonStocks[entry]["symbol"]]=entry



def getBetween(string, before, after):
    return string.split(before)[1].split(after)[0]

def printValues(stock, stocksym, BEP, act_val, exp_val, exp_max, exp_min,  exp_perc, nrecos, nmonths):
    col_ini = CEND
    if   "-" in str(exp_perc) or exp_perc<0 : col_ini = '\033[31m'
    elif "+" in str(exp_perc) or exp_perc>0 : col_ini =  '\033[32m'
    if float(exp_perc)>0: exp_perc = "+"+str(float(exp_perc))
    col_end = CEND
    print stock, "["+stocksym+"]"
    print "current value:", '\033[34m'+act_val+col_end, "\t with BEP:,", '\033[34m'+BEP+col_end
    print "expected value in" ,nmonths, "months:", exp_val, "all", nrecos, "analysis fall within ["+exp_max+","+exp_min+"]"
    print "Current average gain:", col_ini+str(exp_perc)+col_end
    print ""

def makeSoup(link):
    print link
    request   = requests.get(link, headers=headers)
    return  BeautifulSoup(request.text,"lxml")


def getStocks(stocks, type_i):
    for stocksym in stocks.keys():
        json_i = jsonStocks[symb_isin[stocksym]]
        BEP     = str(json_i["BEP"])
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
        print "link", link
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
        json_i["act_val" ] = float(act_val)
        json_i["exp_med" ] = float(exp_med)
        json_i["exp_max" ] = float(exp_max)
        json_i["exp_min" ] = float(exp_min)
        json_i["exp_perc"] = float(exp_perc)
        json_i["nrecos"  ] = float(nrecos)
        json_i["nmonths" ] = float(nmonths)
        json_i["recos"   ] = recos
        
        printValues(stock, stocksym, BEP, act_val, exp_med, exp_max, exp_min, exp_perc, nrecos, nmonths)
        


   
pickleDict = open(foldir+"Portfolio_dict.pkl", "rb")
portfolio  = pickle.load(pickleDict)


pickleDict.close()

getStocks(portfolio, "multiple")
#save to json                                              
with open('act_info.json', 'w') as f:
    json.dump(jsonStocks, f, indent=4)
