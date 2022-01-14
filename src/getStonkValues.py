# -*- coding: utf-8 -*-
from variables import *

currencies = {
    "$"      : "USD",
    u"\u20ac": "EUR", 
    u"\xa5"  : "CNY-JPY"
}
exch = {} #add to dictionary, if not, calculate again

c = CurrencyConverter()




def convertCurrency(input_val, curr_i, curr_f, ref = 0 ):
    value = float(input_val.replace(',',''))
    if float(ref)>0:        
        if "CNYto"+curr_f not in exch : exch["CNYto"+curr_f] = c.convert(1.0, "CNY", curr_f)
        if "JPYto"+curr_f not in exch : exch["JPYto"+curr_f] = c.convert(1.0, "JPY", curr_f)  
        yuan = value*exch["CNYto"+curr_f]
        yen  = value*exch["JPYto"+curr_f]
        if abs(np.log10(yuan/ref)) < abs(np.log10(yen/ref)) : return round(yuan,2)
        else : return round(yen,2)
    else:
        key=curr_i+"to"+curr_f
        if curr_i == curr_f: return round(value,2)
        else:
            if key not in exch: exch[key] = c.convert(1.0, curr_i, curr_f)
            return round(exch[key]*value,2)


def getBetween(string, before, after, curr_i = None, curr_f = None, ref= -1 ):
    split = string.split(before)[1].split(after)[0]
    if curr_i and curr_f: split = convertCurrency(split, curr_i, curr_f, float(ref))
    return split

def printValues(stock, stocksym, BEP, act_val, exp_val, exp_max, exp_min,  exp_perc, nrecos, nmonths, curr_i):
    col_ini = CEND
    if   "-" in str(exp_perc) or exp_perc<0 : col_ini = '\033[31m'
    elif "+" in str(exp_perc) or exp_perc>0 : col_ini =  '\033[32m'
    if float(exp_perc)>0: exp_perc = "+"+str(float(exp_perc))
    col_end = CEND
    print stock, "["+stocksym+"]"
    print "current value:", '\033[34m'+act_val+curr_i+col_end, "\t with BEP:,", '\033[34m'+BEP+curr_i+col_end
    print "expected value in" ,nmonths, "months:", exp_val+curr_i, "all", nrecos, "analysis fall within ["+exp_min+curr_i+","+exp_max+curr_i+"]"
    print "Current expected gain:", col_ini+str(exp_perc)+"%"+col_end
    print ""

def makeSoup(link):
    request = requests.get(link, headers=headers["2"])
    return    BeautifulSoup(request.text,"lxml")


def getStocks(stocks, type_i):
    for stocksym in stocks.keys():
        print stocksym
        if "." in stocksym: continue
        #if "BABA" not in stocksym: continue
        json_i  = jsonStocks[symb_isin[stocksym]]
        BEP     = str(json_i["BEP"])
        curr_i  = str(json_i["currency"])
        symb_i  = currencies.keys()[currencies.values().index(curr_i)]
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
        nmonths   = '12'
        stock     = stocks[stocksym][0]
        reco_info = makeSoup(link)

        if "wsj" in stockType:
            hist_ranP  = reco_info.find(class_="cr_curr_price")
            hist_ran   = reco_info.find(class_="cr_data_collection cr_charts_info").findAll(class_="data_data")
            prices     = reco_info.find(class_="cr_data rr_stockprice module"     ).findAll(class_="data_data")
            reco_table = reco_info.find(class_="cr_analystRatings cr_data module" ).find(class_="cr_dataTable").findAll(class_="data_data")
            print  link
            ran_curr = getBetween(str(hist_ranP), "sup>", "</sup").replace('</','').decode('utf-8') 
            exp_curr = getBetween(str(prices[0]), "sup>", "</sup").replace('</','').decode('utf-8')
            act_curr = getBetween(str(prices[4]), "sup>", "</sup").replace('</','').decode('utf-8')

            ref_hist = -1 
            ref_exp  = -1
            ref_act  = -1
            if "-" in currencies[ran_curr] : ref_hist = getBetween(hist_ranP, "val>", "</span") 
            if "-" in currencies[exp_curr] : ref_exp  = float(getBetween(str(prices[4]  ), "/sup>", "</span").replace(',',''))
            if "-" in currencies[act_curr] : ref_act  = float(getBetween(str(prices[1]  ), "/sup>", "</span").replace(',',''))

            vol_1    = getBetween(str(hist_ran[0]), ">"    , "<"     , currencies[ran_curr], curr_i, ref_hist)
            vol_52   = getBetween(str(hist_ran[1]), ">"    , "<"     , currencies[ran_curr], curr_i, ref_hist)
            ran_1    = getBetween(str(hist_ran[2]), ">"    , "<"     ).split('-')
            ran_52   = getBetween(str(hist_ran[3]), ">"    , "<"     ).split('-')
            min_1    = convertCurrency(ran_1[0] , currencies[ran_curr], curr_i, ref_hist)
            max_1    = convertCurrency(ran_1[1] , currencies[ran_curr], curr_i, ref_hist)
            min_52   = convertCurrency(ran_52[0], currencies[ran_curr], curr_i, ref_hist)
            max_52   = convertCurrency(ran_52[1], currencies[ran_curr], curr_i, ref_hist)
            exp_max  = getBetween(str(prices[0]  ), "/sup>", "</span", currencies[exp_curr], curr_i, ref_exp)
            exp_min  = getBetween(str(prices[2]  ), "/sup>", "</span", currencies[exp_curr], curr_i, ref_exp)
            exp_med  = getBetween(str(prices[1]  ), "/sup>", "</span", currencies[exp_curr], curr_i, ref_exp)
            exp_avg  = getBetween(str(prices[3]  ), "/sup>", "</span", currencies[exp_curr], curr_i, ref_exp)
            act_val  = getBetween(str(prices[4]  ), "/sup>", "</span", currencies[act_curr], curr_i, ref_act)
            
            #exp_perc = round(100*(exp_med-act_val)/act_val,2)
            
            recos    = []
            for i in range(2,len(reco_table),3):
                recos.append(int(getBetween(str(reco_table[i]),">","<")))
            nrecos = sum(recos)

        elif "esp" in stockType:
            dataset     = reco_info.text.split("var barChartData =")[1].split('};')[0]
            histo_table = reco_info.text.split("Tendencia de las recomendaciones")[1].split("*La")[0]
            stock_info  = makeSoup(link.replace('recomendaciones/', '')).find(class_="tabla-contenedor__interior").text.split('<tr>')[0].replace('\n', ' ')

            vol_1    = getBetween(stock_info, "Volumen (acciones)"    , "  ").replace('.','')
            vol_52   = getBetween(stock_info, "Volumen (acciones)"    , "  ").replace('.','')
            max_1    = getBetween(stock_info, u"M\xe1x. intrad\xeda"  , "  ").replace('.','').replace(',','.')
            min_1    = getBetween(stock_info, u"Min. intrad\xeda"     , "  ").replace('.','').replace(',','.')
            max_52   = getBetween(stock_info, u"M\xe1x 52 semanas "   , "  ").replace('.','').replace(',','.')
            min_52   = getBetween(stock_info, u"Min 52 semanas "      , "  ").replace('.','').replace(',','.')
            act_val  = getBetween(stock_info, u"\xdalt. Cotizaci\xf3n", "  ").replace('.','').replace(',','.')
            exp_med  = getBetween(dataset.split('Precio objetivo')[1].split('data')[1], '[', ']').replace("\n\n","").split(',')[0].replace(' ','')
            exp_max  = exp_med
            exp_min  = exp_med
            recos    = histo_table.split("Hoy\n")[1].split("*")[0].split("\n")
            nrecos   = recos[5]
            #exp_perc = round(100*(float(exp_med)-float(act_val))/float(act_val),2)
            
        else:
            print "unrecognised stock Type", stockType
            continue
        reco_wi  = 0
        exp_perc = round(100*(float(exp_med)-float(act_val))/float(act_val),2)
        for i in range(5): reco_wi += float(recos[i])*(i+1)
        json_i["act_val" ] = float(act_val)
        json_i["exp_med" ] = float(exp_med)
        json_i["exp_max" ] = float(exp_max)
        json_i["exp_min" ] = float(exp_min)
        json_i["exp_perc"] = float(exp_perc)
        json_i["nmonths" ] = float(nmonths)
        json_i["max_1"   ] = float(max_1)
        json_i["min_1"   ] = float(min_1)
        json_i["max_52"  ] = float(max_52)
        json_i["min_52"  ] = float(min_52)
        json_i["vol_1"   ] = int  (vol_1)
        json_i["vol_52"  ] = int  (vol_52)
        json_i["nrecos"  ] = int(nrecos)
        json_i["recos"   ] = recos
        json_i["reco_wi" ] = round(reco_wi/int(nrecos),3)
        printValues(stock, stocksym, BEP,str(act_val), str( exp_med), str( exp_max), str( exp_min), str( exp_perc), str( nrecos), str( nmonths), symb_i)
        
if __name__ == '__main__':

    jsonFile   = open(full_port, "rb")
    jsonStocks = json.load(jsonFile)
    symb_isin  = {}
    for entry in jsonStocks:
        symb_isin[jsonStocks[entry]["symbol"]]=entry

    pickleDict = open(port_pkl, "rb")
    portfolio  = pickle.load(pickleDict)


    pickleDict.close()

    getStocks(portfolio, "multiple")
    #save to json                                              
    with open(act_info, 'w') as f:
        json.dump(jsonStocks, f, indent=4)
