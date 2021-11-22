import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, sys, json
foldir   = os.path.dirname(sys.argv[0])
if "/"  in foldir: foldir += "/" 
plotdir  = foldir+"Plots/"
os.system("mkdir -p " + plotdir)


#plot difference
def diff_plots(act_val, exp_val,BEP,ran_52, typedif):
    plt.gcf().subplots_adjust(bottom=0.15)
    act_plot = plt.errorbar(stocknm, act_val,yerr = ran_52 , marker="d", color='r', label="Stock. "+today, uplims=True, lolims=True, fmt = '.')
    act_plot[-1][0].set_linestyle(':')
    act_plot[-1][1].set_linestyle(':')

    if "abs" in typedif.lower() : gain = "Prize [USD]"    
    else                        : gain = "Gain/loss [%]"

    if "bep" in typedif.lower() or "" in typedif.lower(): plt.scatter(stocknm, BEP    , marker="_",label="Break even point", color='blue', s=80)
    exp_plot = plt.errorbar(stocknm, exp_val, yerr =exp_ran, fmt='.', color='black', label="Exp. "+gain)
    exp_plot[-1][0].set_linestyle('--')
    plt.ylabel(gain)
    plt.xlabel("Stock name")
    plt.title(typedif+" expected gain/loss")
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.savefig(plotdir+typedif.replace(" ","-").lower()+"_difference.png")
    plt.clf()

#Plot recommendation plots
def makerecoplots(sell, underw, hold, overw, buy):
    colrec  = ["red", "orange", "yellow", "yellowgreen", "green"]
    reconms = ["sell", "underweight", "hold", "overwight", "buy" ]
    order   = [4,3,2,1,0]
    plt.bar(stocknm,sell  , color=colrec[0], label=reconms[0])
    plt.bar(stocknm,underw, color=colrec[1], label=reconms[1], bottom=sell)
    plt.bar(stocknm,hold  , color=colrec[2], label=reconms[2], bottom=(sell+underw))
    plt.bar(stocknm,overw , color=colrec[3], label=reconms[3], bottom=sell+underw+hold)
    plt.bar(stocknm,buy   , color=colrec[4], label=reconms[4], bottom=sell+underw+hold+overw)

    plt.title("Expert recommendations")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.xlabel("company")


def gainlossplots(gainloss, plot_type):
    if 'rel' in plot_type: l_type='[%]'
    else: l_type = '[EUR]'
    
    plt.axhline(0, color='black', linestyle='-')
    plt.bar(stocknm, gainloss, color= cgainloss)
    plt.xlabel('Stock name')
    plt.ylabel('Gain/loss '+l_type)
    plt.title(plot_type+' Gain/loss per stock')
    plt.savefig(plotdir+plot_type.lower()+"_gain-loss.png")
    plt.clf()

today    = str(datetime.date(datetime.now()))#-timedelta(days=1)) 

if __name__ == '__main__':
    stocknm = []

    jsonDict  = open("act_info.json", "rb")
    portfolio = json.load(jsonDict)

    stocknm  = np.array([])
    act_val  = np.array([])
    exp_val  = np.array([])
    exp_max  = np.array([])
    exp_min  = np.array([])
    BEP      = np.array([])
    gainloss = np.array([])
    size     = np.array([])

    max_1    = np.array([])
    max_52   = np.array([])
    min_1    = np.array([])
    min_52   = np.array([])

    buy    = np.array([])
    overw  = np.array([])
    hold   = np.array([])
    underw = np.array([])
    sell   = np.array([])
    nrecos = np.array([])

    for isin in portfolio:
        if "ETF" in portfolio[isin]["productType"]:
            ETF_info = portfolio[isin]["productType"]
            continue 
        stocknm  = np.append(stocknm , portfolio[isin][u'symbol'])
        act_val  = np.append(act_val , portfolio[isin][u'act_val'])
        exp_val  = np.append(exp_val , portfolio[isin][u'exp_med'])
        exp_max  = np.append(exp_max , portfolio[isin][u'exp_max'])
        exp_min  = np.append(exp_min , portfolio[isin][u'exp_min'])
        BEP      = np.append(BEP     , portfolio[isin][u'BEP'])
        gainloss = np.append(gainloss, portfolio[isin][u'gain/loss'])
        size     = np.append(size    , portfolio[isin][u'size'])

        max_1    = np.append(max_1   , portfolio[isin][u'max_1'])
        max_52   = np.append(max_52  , portfolio[isin][u'max_52'])
        min_1    = np.append(max_1   , portfolio[isin][u'min_1'])
        min_52   = np.append(min_52  , portfolio[isin][u'min_52'])

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

    colors     = [ 'r' if i < 0 else 'g' for i in difference]
    cgainloss  = [ 'r' if i < 0 else 'g' for i in gainloss]


    gainlossplots(gainloss/(0.01*BEP*size), 'Relative')
    gainlossplots(gainloss                , 'Total')
    ran_52 = [act_val-min_52, max_52-act_val]
    relran_52 = 100*(ran_52/act_val)
    BEPran_52 = 100*(ran_52/BEP)


    diff_plots(act_val   , exp_val   , BEP   ,ran_52   , "Absolute")
    diff_plots(act_relval, exp_relval, BEP   ,relran_52, "Relative")
    diff_plots(act_relBEP, exp_relBEP, relBEP,BEPran_52, "relative BEP")


    makerecoplots(sell, underw, hold, overw, buy)
    makerecoplots(100*sell/nrecos, 100* underw/nrecos, 100*hold/nrecos, 100*overw/nrecos, 100*buy/nrecos)
