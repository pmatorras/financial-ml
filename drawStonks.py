import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
stocks = pd.read_csv("stocks.txt", sep="\t")
now   = datetime.now()
today = str(datetime.date(now))

stocknms = []
for stock in  stocks["abrv"]:
    if stock not in stocknms: stocknms.append(stock)

print stocks.columns

print bool('act_val' == stocks["variable"][0])
print  stocks[stocks["variable"] == "act_val"] 
#print [stocks["variable"]=="act_val"]

actval = stocks[stocks["variable"]=="act_val"][str(today)]
expval = stocks[stocks["variable"]=="exp_val"][str(today)]
expmax = stocks[stocks["variable"]=="exp_max"][str(today)]
expmin = stocks[stocks["variable"]=="exp_min"][str(today)]
#expran = [expval-expmin,expmax-expval]
expminrel = expval.reset_index()-expmin.reset_index()
expmaxrel = expmax.reset_index()-expval.reset_index()

expran = np.array(list(zip(expminrel[str(today)], expmaxrel[str(today)]))).T

difference = actval.reset_index() -expval.reset_index()
colors = []
print difference
for dif in difference[today]:
    if dif<0: colors.append('g')
    else: colors.append('r')

plt.gcf().subplots_adjust(bottom=0.15)
plt.scatter(stocknms, actval, marker="_",label="Stock value at "+str(today), color=colors)
plt.errorbar(stocknms, expval, yerr =expran, fmt='.', color='blue', label="expected value")
plt.ylabel("Prize [USD]")
plt.xlabel("Stock name")
plt.title("Stock prize vs forecast")
plt.legend()
plt.tick_params(axis='x', rotation=45)
plt.savefig("absolute_difference.pdf")

plt.clear()
