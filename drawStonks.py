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
    print  stocknm, sell
    plt.bar(stocknm,sell  , color=colrec[0], label=reconms[0])
    plt.bar(stocknm,underw, color=colrec[1], label=reconms[1], bottom=sell)
    plt.bar(stocknm,hold  , color=colrec[2], label=reconms[2], bottom=(sell+underw))
    plt.bar(stocknm,overw , color=colrec[3], label=reconms[3], bottom=sell+underw+hold)
    plt.bar(stocknm,buy   , color=colrec[4], label=reconms[4], bottom=sell+underw+hold+overw)

    plt.title("Expert recommendations")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.xlabel("company")

#plot difference
def diff_plots(act_val, exp_val,BEP,typedif):
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.scatter(stocknm, act_val, marker="X",label="Stock value at "+today, color=colors)
    if "abs" in typedif.lower() : gain = "Prize [USD]"    
    else                        : gain = "Gain/loss [%]"

    if "bep" in typedif.lower() or "abs" in typedif.lower(): plt.scatter(stocknm, BEP    , marker="_",label="Break even point", color='blue')

    plt.errorbar(stocknm, exp_val, yerr =exp_ran, fmt='.', color='black', label="Exp. "+gain)
    plt.ylabel(gain)
    plt.xlabel("Stock name")
    plt.title(typedif+" expected gain/loss")
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.savefig(plotdir+typedif.replace(" ","-")+"_difference.pdf")
    plt.clf()


today    = str(datetime.date(datetime.now()))#-timedelta(days=1)) 
stocknm = []

jsonDict  = open("act_info.json", "rb")
portfolio = json.load(jsonDict)

#print (portfolio), (stocks)
stocknm = np.array([])
act_val = np.array([])
exp_val = np.array([])
exp_max = np.array([])
exp_min = np.array([])
BEP     = np.array([])

buy    = np.array([])
overw  = np.array([])
hold   = np.array([])
underw = np.array([])
sell   = np.array([])
nrecos = np.array([])
for isin in portfolio:
    if "ETF" in portfolio[isin]["productType"]: continue 
    stocknm = np.append(stocknm,portfolio[isin][u'symbol'])
    act_val = np.append(act_val,portfolio[isin][u'act_val'])
    exp_val = np.append(exp_val,portfolio[isin][u'exp_med'])
    exp_max = np.append(exp_max,portfolio[isin][u'exp_max'])
    exp_min = np.append(exp_min,portfolio[isin][u'exp_min'])
    BEP     = np.append(BEP    ,portfolio[isin][u'BEP'])

    recos  = portfolio[isin][u'recos']
    nrecos = np.append(nrecos,portfolio[isin][u'nrecos'])
    buy    = np.append(buy   , int( recos[0]))
    overw  = np.append(overw , int( recos[1]))
    hold   = np.append(hold  , int( recos[2]))
    underw = np.append(underw, int( recos[3]))
    sell   = np.append(sell  , int( recos[4]))

#get differences
exp_mindif    = exp_val-exp_min
exp_maxdif    = exp_max-exp_val
difference    = act_val-exp_val
act_relval    = -100 + 100*act_val/act_val
exp_relval    = -100 + 100*(exp_val/act_val)
relBEP        = -100 + 100*BEP/BEP
exp_relBEP    = -100 + 100*(exp_val/BEP)
act_relBEP    = -100 + 100*(act_val/BEP)
exp_minreldif = 100*exp_mindif/act_val
exp_maxreldif = 100*exp_maxdif/act_val
exp_relran    = np.array(list(zip(exp_minreldif, exp_maxreldif))).T
exp_ran       = np.array(zip(exp_mindif, exp_maxdif)).T
print "Range",exp_ran
colors     = []
for dif in difference:
    if dif<0: colors.append('g')
    else    : colors.append('r')

diff_plots(act_val   , exp_val   , BEP, "absolute")
diff_plots(act_relval, exp_relval, BEP,"relative")
diff_plots(act_relBEP, exp_relBEP, relBEP,"relative BEP")


makerecoplots(sell, underw, hold, overw, buy)
plt.ylabel("n recommendations")
plt.savefig(plotdir+"tot_recommendations.png")
plt.clf()
makerecoplots(100*sell/nrecos, 100* underw/nrecos, 100*hold/nrecos, 100*overw/nrecos, 100*buy/nrecos)
plt.ylabel("recommendations [%]")
plt.savefig(plotdir+"rel_recommendations.png")
plt.clf()    
    


        


