import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, sys, json


foldir   = os.path.dirname(sys.argv[0])
stocks   = pd.read_csv("stocks.txt", sep="\t")
now      = datetime.now()
today    = str(datetime.date(now))#-timedelta(days=1)) 
stocknms = []

jsonDict  = open("full_portfolio.json", "rb")
portfolio = json.load(jsonDict)

#print stocks['stock']
#print "portfolio", portfolio[0]
#for key in portfolio:
#    print key, portfolio[key]

#exit()

#print stocks.keys()
if "/"  in foldir: foldir += "/" 
plotdir  = foldir+"Plots/"
os.system("mkdir -p " + plotdir)
for stock in  stocks["symb"]:
    if stock not in stocknms: stocknms.append(stock)

#get abs/rel difference info
act_val    = stocks[stocks["variable"]=="act_val"][today]
exp_val    = stocks[stocks["variable"]=="exp_val"][today]
exp_max    = stocks[stocks["variable"]=="exp_max"][today]
exp_min    = stocks[stocks["variable"]=="exp_min"][today]
exp_mindif = exp_val.reset_index(drop=True)-exp_min.reset_index(drop=True)
exp_maxdif = exp_max.reset_index(drop=True)-exp_val.reset_index(drop=True)
exp_ran    = np.array(list(zip(exp_mindif, exp_maxdif))).T

act_relval    = -100 + 100*act_val/act_val
exp_relval    = -100 + 100*(exp_val.reset_index(drop=True)/act_val.reset_index(drop=True))
exp_minreldif = 100*exp_mindif.reset_index(drop=True)/act_val.reset_index(drop=True)
exp_maxreldif = 100*exp_maxdif.reset_index(drop=True)/act_val.reset_index(drop=True)
exp_relran    = np.array(list(zip(exp_minreldif, exp_maxreldif))).T

#Get sell info
buy    = stocks[stocks["variable"]=="buy"    ][today].reset_index(drop=True)
overw  = stocks[stocks["variable"]=="overw"  ][today].reset_index(drop=True)
hold   = stocks[stocks["variable"]=="hold"   ][today].reset_index(drop=True)
underw = stocks[stocks["variable"]=="underw" ][today].reset_index(drop=True)
sell   = stocks[stocks["variable"]=="sell"   ][today].reset_index(drop=True)
nrecom = stocks[stocks["variable"]=="n_recom"][today].reset_index(drop=True)
#allrec = buy+overw+hold+underw+sell

#print len(buy), len(overw), len(hold), len(underw), len(sell)
allrec  = np.vstack([buy,overw])#, hold, underw, sell])#,hold,underw,sell])
colrec  = ["red", "orange", "yellow", "yellowgreen", "green"]
reconms = ["sell", "underweight", "hold", "overwight", "buy" ]
order   = [4,3,2,1,0]

def makerecoplots(sell, underw, hold, overw, buy):
    plt.bar(stocknms,sell  , color=colrec[0], label=reconms[0])
    plt.bar(stocknms,underw, color=colrec[1], label=reconms[1], bottom=sell)
    plt.bar(stocknms,hold  , color=colrec[2], label=reconms[2], bottom=(sell+underw))
    plt.bar(stocknms,overw , color=colrec[3], label=reconms[3], bottom=sell+underw+hold)
    plt.bar(stocknms,buy   , color=colrec[4], label=reconms[4], bottom=sell+underw+hold+overw)

    plt.title("Expert recommendations")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.xlabel("company")
makerecoplots(sell, underw, hold, overw, buy)
plt.ylabel("n recommendations")

plt.savefig(plotdir+"tot_recommendations.png")
plt.clf()

makerecoplots(100*sell/nrecom, 100* underw/nrecom, 100*hold/nrecom, 100*overw/nrecom, 100*buy/nrecom)
plt.ylabel("recommendations [%]")

plt.savefig(plotdir+"rel_recommendations.png")
plt.clf()    
    

difference = act_val.reset_index() -exp_val.reset_index()
colors     = []
longcolors = []
for dif in difference[today]:
    if dif<0:
        colors.append('g')
        longcolors.append([0,1,0,1])
    else:
        colors.append('r')
        longcolors.append([1,0,0,1])

#absolute difference
plt.gcf().subplots_adjust(bottom=0.15)
plt.scatter(stocknms, act_val, marker="_",label="Stock value at "+today, color=colors)
plt.errorbar(stocknms, exp_val, yerr =exp_ran, fmt='.', color='blue', label="expected value")
plt.ylabel("Prize [USD]")
plt.xlabel("Stock name")
plt.title("Stock prize vs forecast")
plt.legend()
plt.tick_params(axis='x', rotation=45)
plt.savefig(plotdir+"absolute_difference.pdf")

plt.clf()

#Relative difference
plt.scatter(stocknms, act_relval, marker="_",label="Stock value at "+today, color=colors)
plt.errorbar(stocknms, exp_relval, yerr =exp_relran, color='blue', label="expected value", linestyle="none", marker ="s")

plt.ylabel("Exp. gain [%]")
plt.xlabel("Stock name")
plt.title("Relative expected gain/loss")
plt.legend()
plt.tick_params(axis='x', rotation=45)
plt.savefig(plotdir+"relative_difference.pdf")
