import requests, json, re, csv, os
from bs4 import BeautifulSoup
from datetime import datetime
now = datetime.now()
today = datetime.date(now)
def getBetween(string, before, after):
    return string.split(before)[1].split(after)[0]

def writeCSV(csvname,stock, stocksym, act_val, exp_val, exp_max, exp_min, nrecos):
    csvexist= os.path.exists(csvname)
    with open(csvname, 'a') as f:
        writer   = csv.writer(f, delimiter = '\t')
        titlerow = stock+"\t["+stocksym+"]\t" 
        if csvexist is False : writer.writerow(['stock','abrv' , 'variable', str(today)])
        writer.writerow([stock, stocksym, "act_val", act_val])
        writer.writerow([stock, stocksym, "exp_val", exp_val])
        writer.writerow([stock, stocksym, "exp_max", exp_max])
        writer.writerow([stock, stocksym, "exp_min", exp_min])
        writer.writerow([stock, stocksym, "n_recom", nrecos ])
        writer.writerow("")

CRED = '\033[91m'
CEND = '\033[0m'


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
csvname= 'stocks.txt'
os.system('rm '+csvname)


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'}
stocks_wsj = {"PROSUS N.V." : ["PRX", "NL/XAMS/"], "Volkswagen" : ["VOW3", "XE/XETR/"]}

def makeSoup(link):
    request   = requests.get(link, headers=headers)
    return  BeautifulSoup(request.text,"lxml")
    #return  BeautifulSoup(request.text,'html.parser')



for stock in stocks_wsj.keys():
    link = "https://www.wsj.com/market-data/quotes/"+stocks_wsj[stock][1]+stocks_wsj[stock][0]+"/research-ratings"
    soup = makeSoup(link)
    stocksym   = stocks_wsj[stock][0]
    reco_table =  soup.find(class_="cr_analystRatings cr_data module").find(class_="cr_dataTable")
    buy    = str(reco_table.findAll(class_="data_data")[2 ]).split(">")[1].split("<")[0]
    overw  = str(reco_table.findAll(class_="data_data")[5 ]).split(">")[1].split("<")[0]
    hold   = str(reco_table.findAll(class_="data_data")[8 ]).split(">")[1].split("<")[0]
    underw = str(reco_table.findAll(class_="data_data")[11]).split(">")[1].split("<")[0]
    sell   = str(reco_table.findAll(class_="data_data")[14]).split(">")[1].split("<")[0]
    prices = soup.find(class_="cr_data rr_stockprice module").findAll(class_="data_data")

    exp_max  = getBetween(str(prices[0]), "/sup>", "</span")
    exp_min  = getBetween(str(prices[2]), "/sup>", "</span")
    exp_med  = getBetween(str(prices[1]), "/sup>", "</span")
    exp_avg  = getBetween(str(prices[3]), "/sup>", "</span")
    act_val  = getBetween(str(prices[4]), "/sup>", "</span")
    exp_perc = round(100*(float(exp_med)-float(act_val))/float(act_val),2)
    nrecos   = int(buy)+int(overw)+int(hold)+int(underw)+int(sell)
    nmonths  = '12'

    writeCSV(csvname, stock, stocksym, act_val, exp_med, exp_max, exp_min, nrecos)
    printValues(stock, stocksym, act_val, exp_med, exp_max, exp_min, exp_perc, nrecos, nmonths)


stocks_esp = {"LOGISTA" : ["logista_hlgd_sa","56747"], "ORYZON GENOMICS" : ["oryzon_genomics_sa", "57000"] , "GRENERGY RENOVABLES" : ["grenergy_renovables_sa","56988"]}

for stock in stocks_esp.keys():
    link      = "https://cincodias.elpais.com/mercados/empresas/"+stocks_esp[stock][0]+"/"+stocks_esp[stock][1]+"/recomendaciones/"
    request   = requests.get(link)
    soup      = BeautifulSoup(request.text,"lxml")
    dataset   = soup.text.split("var barChartData =")[1].split('};')[0]
    recommend =  soup.text.split("Tendencia de las recomendaciones")[1].split("*La")[0]
    act_val   = getBetween(dataset.split('Precio real')[1].split('data')[1], '[', ']')
    exp_val   = getBetween(dataset.split('Precio objetivo')[1].split('data')[1], '[', ']')
    today_exp_val   = exp_val.replace("\n\n","").split(',')[1]
    today_act_val   = act_val.replace("\n\n","").split(',')[1]
    today_recommend = recommend.split("Hoy\n")[1].split("*")[0].split("\n")
    nrecos   = today_recommend[5]
    exp_perc = round(100*(float(today_exp_val)-float(today_act_val))/float(today_act_val),2)

    printValues(stock, stock, today_act_val, today_exp_val, today_exp_val, today_exp_val, exp_perc, nrecos, nmonths)
    writeCSV(csvname, stock, stock.split(' ')[0], today_act_val, today_exp_val, today_exp_val, today_exp_val, nrecos)

stocks_cnn = {"Alibaba" : "BABA", "Airbus" : "EADSY", "Curevac"   : "CVAC",
              "Arcelor" : "MT"  , "TSMC  " : "TSM"  , "Dr Horton" : "DHI" ,
              "Total"   : "TOT"}
linkbase = "https://money.cnn.com/quote/forecast/forecast.html?symb="

for stock in stocks_cnn.keys():
    stocksym  = stocks_cnn[stock]
    link      = linkbase+stocksym
    request   = requests.get(link)
    soup      = BeautifulSoup(request.text,"lxml")
    valheader = soup.find(class_='wsod_last')


    act_val = getBetween(str(valheader), '"ToHundredth">', "</span")
    name = str(soup.find(class_="wsod_fLeft wsod_narrowH1Container"))
    #print soup.find_all('p')


    forecast= str(soup.find(class_='wsod_twoCol clearfix'))
    numbers =  re.findall(r"[-+]?\d*\.\d+|\d+", forecast)

    nrecos   = numbers[0]
    nmonths  = numbers[1]
    exp_val  = numbers[2]
    exp_max  = numbers[3]
    exp_min  = numbers[4]
    exp_perc = numbers[5]
    act_val2 = numbers[6]

    col_ini = CEND
    if   "-" in exp_perc : col_ini = '\033[31m'
    elif "+" in exp_perc : col_ini =  '\033[32m'
    col_end = CEND
    
    
    if float(act_val) != float(act_val2):
        print CRED + "Error, numbers "+act_val+" and "+act_val2+" are different!" + CEND
 
    printValues(stock, stocksym, act_val, exp_val, exp_max, exp_min, exp_perc, nrecos, nmonths)
    writeCSV(csvname, stock, stocksym, act_val, exp_val, exp_max, exp_min, nrecos)
