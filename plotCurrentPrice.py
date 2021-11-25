from drawStonks      import *
from pandas.plotting import register_matplotlib_converters
import yfinance         as yf
import matplotlib.dates as mdates
register_matplotlib_converters()
import plotly.graph_objects as go

jsonDict  = open("act_info.json", "rb")
portfolio = json.load(jsonDict)
keys      = ''
stockpdir = plotdir+"Stocks/"
stocknm   = []
currency  = []
os.system('mkdir -p '+ stockpdir) 
stockinfo = {}
for stock_id in portfolio:
    symbol  = portfolio[stock_id][u'symbol']
    isin    = portfolio[stock_id][u'isin']
    stocknm. append(portfolio[stock_id][u'name'])
    currency.append(portfolio[stock_id][u'currency'])
    ext     = ''
    if "NL"  in isin  : ext = ".AS"
    if "DE"  in isin  : ext = ".DE"
    if "ES"  in isin  : ext = ".MC"
    if "IE"  in isin  : ext = ".DE"
    if "AIR" in symbol: ext = ".DE" #AD-HOC solution, may need to be fixed in the future  
    keys += symbol+ext+' '
    print symbol, isin
    stockinfo[symbol+ext] = {"isin" : isin, "name": portfolio[stock_id][u'name'], "currency" : portfolio[stock_id][u'currency']}

#keys    = "VOW3.DE PRX.AS"
periods = ["ytd"]

for period in periods:
    data = yf.download(keys, period=period)
    old  = data.reset_index()
    print old.keys()
    for idx, key, in enumerate(keys.strip().split(' ')):
        fignm = stockpdir+key.replace('.','-')+"_"+period+".png"
        fig = go.Figure(data  = [go.Candlestick(x     = old['Date'],
                                                open  = old['Open'][key],
                                                high  = old['High'][key],
                                                low   = old['Low'][key],
                                                close = old['Close'][key])])

        fig.update_layout(
            title       = "Stock price, "+stockinfo[key]["name"]+" ("+period+")",
            yaxis_title = key+' Stock')
        fig.write_image(fignm)
        print "Plotting", fignm
        #fig.show()
