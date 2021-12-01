import plotly.graph_objects as go
import yfinance             as yf
from   pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def onlySamples(doOnly, onlySymbs, stock_sym):
    if doOnly:
        isHere = False
        for onlysymb in onlySymbs:
            if onlysymb in stock_sym: isHere = True
        return isHere
    else:
        return True

if __name__== '__main__':
    jsonDict  = open(act_info, "rb")
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
        isHere  = onlySamples(doOnly,onlySymbs, symbol)
        if isHere is False: continue
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

    period_info = {"1d" : { "interval" : "5m"  , "Date" : "Datetime"}, "5d" : { "interval" : "30m", "Date" : "Datetime" },
                   "1mo": { "interval" : "1d"  , "Date" : "Date"    }, "3mo": { "interval" : "1d" , "Date" : "Date" },
                   "6mo": { "interval" : "1d"  , "Date" : "Date"    }, "ytd": { "interval" : "1d" , "Date" : "Date" }, 
                   "1y" : { "interval" : "1wk" , "Date" : "Date"    }, "2y" : { "interval" : "1wk", "Date" : "Date" },
                   "5y" : { "interval" : "1mo" , "Date" : "Date"    }, "10y": { "interval" : "3mo", "Date" : "Date" },
                   "max": { "interval" : "3mo", "Date" : "Date"    } }
    
    if opt.period:
        if "all" in opt.period: periods= period_info #valid_periods
        else:
            periods = opt.period.split('_')
    else:
        print "in else"
        periods = ["ytd"]

    #make plotting
    for period in periods:
        if period not in period_info.keys():
            print "invalid period", period
            print "Please choose one of these:", period_info.keys()
            continue

        data = yf.download(keys, period=period, interval=period_info[period]["interval"]).reset_index()
        for idx, key, in enumerate(keys.strip().split(' ')):
            fignm = stockpdir+key.replace('.','-')+"_"+period+".png"
            fig   = go.Figure(data  = [go.Candlestick(x     = data[period_info[period]["Date"]],
                                                      open  = data['Open'][key],
                                                      high  = data['High'][key],
                                                      low   = data['Low'][key],
                                                      close = data['Close'][key])])

            fig.update_layout(
                height = 500,
                title  = dict(
                    text = "<b>Stock price, "+stockinfo[key]["name"]+" ("+period+")</b>",
                    x    = 0.5,
                    y    = 0.95,
                    font = dict(
                        family = "Arial",
                        size   = 26,
                        color  = '#000000'
                    )
                ),
                yaxis_title = '<b>'+key+' Stock</b>',
                font        = dict(
                    family = "Courier New, Monospace",
                    size   = 16,
                    color  = '#000000'
                )
            )

            fig.write_image(fignm, scale=1, height=900, width=1200 )
            print "Plotting", fignm
            if opt.interact: fig.show()
