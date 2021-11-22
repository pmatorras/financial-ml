import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os,sys
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
foldir   = os.path.dirname(sys.argv[0])
plotdir  = foldir+"Plots/"



keys= "VOW3.DE PRX.AS"
data = yf.download(keys, period="ytd")



for key in keys.split(' '):
    print key
    plt.plot(data["Close"][key])
    plt.ylabel("Stock prize")
    plt.xlabel("Day")
    plt.title("stock"+key)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    plt.savefig(plotdir+key.replace('.','_')+".png")
    plt.clf()
