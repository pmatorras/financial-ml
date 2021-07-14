import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys


foldir   = os.path.dirname(sys.argv[0])
stocks   = pd.read_csv("stocks.txt", sep="\t")
now      = datetime.now()
today    = str(datetime.date(now))
stocknms = []

if "/"  in foldir: foldir += "/" 
plotdir  = foldir+"Plots/"
os.system("mkdir -p " + plotdir)
for stock in  stocks["abrv"]:
    if stock not in stocknms: stocknms.append(stock)


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

#Get colours
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
