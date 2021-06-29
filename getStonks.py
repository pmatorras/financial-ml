import requests, json, re, csv, os
from bs4 import BeautifulSoup
from datetime import datetime
now = datetime.now()
today = datetime.date(now)
def getBetween(string, before, after):
    return string.split(before)[1].split(after)[0]


CRED = '\033[91m'
CEND = '\033[0m'
csvname= 'stocks.txt'
os.system('rm '+csvname)
    
stocks = {"Alibaba" : "BABA", "Airbus" : "EADSY", "Curevac"   : "CVAC",
          "Arcelor" : "MT"  , "TSMC  " : "TSM"  , "Dr Horton" : "DHI" ,
          "Total"   : "TOT"}
linkbase = "https://money.cnn.com/quote/forecast/forecast.html?symb="

for stock in stocks.keys():
    stocksym = stocks[stock]
    print stock, "["+stocksym+"]"
    link = linkbase+stocksym
    r2 = requests.get(link)

    soup = BeautifulSoup(r2.text,"lxml")
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


    if float(act_val) != float(act_val2):
        print CRED + "Error, numbers "+act_val+" and "+act_val2+" are different!" + CEND

    print "current value:", act_val
    print "expected value in "+nmonths+" months:", exp_val, "all "+nrecos+" analysis fall within ["+exp_max+","+exp_min+"]"
    print "Current average gain:", exp_perc


    csvexist= os.path.exists(csvname)
    with open(csvname, 'a') as f:
        writer   = csv.writer(f, delimiter = '\t')
        titlerow = stock+"\t["+stocksym+"]\t" 
        if csvexist is False : writer.writerow(['stock','abrv' , 'variable', str(today)])
        writer.writerow([stock, stocksym, "act_val ", act_val])
        writer.writerow([stock, stocksym, "exp_val ", exp_val])
        writer.writerow([stock, stocksym, "exp_min ", exp_max])
        writer.writerow([stock, stocksym, "exp_max ", exp_min])
        writer.writerow([stock, stocksym, "n_recom ", nrecos ])
        writer.writerow("")
        #writer.writerows(titlerow+act_val)
        #writer.writerows("\n")
    #exit()
'''

nreco = getBetween(forecast, "The", "analysts")
median = getBetween(forecast,"median target of", ", with")
high   = getBetween(forecast,"high estimate of", "and a ")
low    = getBetween(forecast,"low estimate of" , ". The ")
print nreco
'''
