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


act_val = stocks[stocks["variable"]=="act_val"][today]
exp_val = stocks[stocks["variable"]=="exp_val"][today]
exp_max = stocks[stocks["variable"]=="exp_max"][today]
exp_min = stocks[stocks["variable"]=="exp_min"][today]

act_relval    = 100*act_val/act_val
exp_relval    = 100*(exp_val.reset_index()/act_val.reset_index())[today]
exp_mindif    = exp_val.reset_index()-exp_min.reset_index()
exp_maxdif    = exp_max.reset_index()-exp_val.reset_index()
exp_minreldif = 100*exp_mindif.reset_index()/act_val.reset_index()
exp_maxreldif = 100*exp_maxdif.reset_index()/act_val.reset_index()

exp_ran    = np.array(list(zip(exp_mindif[today], exp_maxdif[today]))).T
exp_relran = np.array(list(zip(exp_minreldif[today], exp_maxreldif[today]))).T

difference = act_val.reset_index() -exp_val.reset_index()
colors     = []

for dif in difference[today]:
    if dif<0: colors.append('g')
    else: colors.append('r')

plt.gcf().subplots_adjust(bottom=0.15)
plt.scatter(stocknms, act_val, marker="_",label="Stock value at "+today, color=colors)
plt.errorbar(stocknms, exp_val, yerr =exp_ran, fmt='.', color='blue', label="expected value")
plt.ylabel("Prize [USD]")
plt.xlabel("Stock name")
plt.title("Stock prize vs forecast")
plt.legend()
plt.tick_params(axis='x', rotation=45)
plt.savefig("absolute_difference.pdf")

plt.clf()

print exp_relval, "\n neew", stocknms
plt.scatter(stocknms, act_relval, marker="_",label="Stock value at "+today, color=colors)
plt.errorbar(stocknms, exp_relval, yerr =exp_relran, fmt='.', color='blue', label="expected value")
plt.ylabel("Percentage change")
plt.xlabel("Stock name")
plt.title("Relative expected gain/loss")
plt.legend()
plt.tick_params(axis='x', rotation=45)
plt.savefig("relative_difference.pdf")
