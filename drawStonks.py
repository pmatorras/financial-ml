import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, sys, json
foldir   = os.path.dirname(sys.argv[0])
if "/"  in foldir: foldir += "/" 
plotdir  = foldir+"Plots/"
os.system("mkdir -p " + plotdir)

def makerecoplots(sell, underw, hold, overw, buy):
    colrec  = ["red", "orange", "yellow", "yellowgreen", "green"]
    reconms = ["sell", "underweight", "hold", "overwight", "buy" ]
    order   = [4,3,2,1,0]
    plt.bar(stocknms,sell  , color=colrec[0], label=reconms[0])
    plt.bar(stocknms,underw, color=colrec[1], label=reconms[1], bottom=sell)
    plt.bar(stocknms,hold  , color=colrec[2], label=reconms[2], bottom=(sell+underw))
    plt.bar(stocknms,overw , color=colrec[3], label=reconms[3], bottom=sell+underw+hold)
    plt.bar(stocknms,buy   , color=colrec[4], label=reconms[4], bottom=sell+underw+hold+overw)

    plt.title("Expert recommendations")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.xlabel("company")

#plot difference
def diff_plots(act_val, exp_val,stocknms,typedif):
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.scatter(stocknms, act_val, marker="_",label="Stock value at "+today, color=colors)
    if "abs" in typedif.lower() : gain = "Prize [USD]"
    else                        : gain = "Gain/loss [%]"
    plt.errorbar(stocknms, exp_val, yerr =exp_ran, fmt='.', color='blue', label="Exp. "+gain)
    plt.ylabel(gain)
    plt.xlabel("Stock name")
    plt.title(typedif+" expected gain/loss")
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.savefig(plotdir+typedif+"_difference.pdf")
    plt.clf()


stocks   = pd.read_csv("stocks.txt", sep="\t")
now      = datetime.now()
today    = str(datetime.date(now))#-timedelta(days=1)) 
stocknms = []

jsonDict  = open("act_info.json", "rb")
portfolio = json.load(jsonDict)

#print (portfolio), (stocks)
stocknm = np.array([])
act_val = np.array([])
exp_val = np.array([])
exp_max = np.array([])
exp_min = np.array([])
for isin in portfolio:
    if "ETF" in portfolio[isin]["productType"]: continue 
    stocknm = np.append(stocknm,portfolio[isin][u'symbol'])
    act_val = np.append(act_val,float(portfolio[isin][u'act_val']))
    exp_val = np.append(exp_val,float(portfolio[isin][u'exp_med']))
    exp_max = np.append(exp_max,float(portfolio[isin][u'exp_max']))
    exp_min = np.append(exp_min,float(portfolio[isin][u'exp_min']))

exp_mindif    = exp_val-exp_min
exp_maxdif    = exp_max-exp_val
difference    = act_val-exp_val
act_relval    = -100 + 100*act_val/act_val
exp_relval    = -100 + 100*(exp_val/act_val)
exp_minreldif = 100*exp_mindif/act_val
exp_maxreldif = 100*exp_maxdif/act_val
exp_relran    = np.array(list(zip(exp_minreldif, exp_maxreldif))).T

print exp_mindif,"\n", exp_maxdif
exp_ran    = np.array(zip(exp_mindif, exp_maxdif)).T
print "Range",exp_ran
colors     = []
for dif in difference:
    if dif<0: colors.append('g')
    else    : colors.append('r')
        
diff_plots(act_val   , exp_val   , stocknm, "absolute")
diff_plots(act_relval, exp_relval, stocknm,"relative")

for stock in  stocks["symb"]:
    if stock not in stocknms: stocknms.append(stock)



#Get sell info
buy    = stocks[stocks["variable"]=="buy"    ][today].reset_index(drop=True)
overw  = stocks[stocks["variable"]=="overw"  ][today].reset_index(drop=True)
hold   = stocks[stocks["variable"]=="hold"   ][today].reset_index(drop=True)
underw = stocks[stocks["variable"]=="underw" ][today].reset_index(drop=True)
sell   = stocks[stocks["variable"]=="sell"   ][today].reset_index(drop=True)
nrecom = stocks[stocks["variable"]=="n_recom"][today].reset_index(drop=True)


makerecoplots(sell, underw, hold, overw, buy)
plt.ylabel("n recommendations")
plt.savefig(plotdir+"tot_recommendations.png")
plt.clf()

makerecoplots(100*sell/nrecom, 100* underw/nrecom, 100*hold/nrecom, 100*overw/nrecom, 100*buy/nrecom)
plt.ylabel("recommendations [%]")
plt.savefig(plotdir+"rel_recommendations.png")
plt.clf()    
    


        


