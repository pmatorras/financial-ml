
#plot difference
def diff_plots(act_val, exp_val,BEP,ran_52, typedif):
    plt.gcf().subplots_adjust(bottom=0.15)
    act_plot = plt.errorbar(stocknm, act_val,yerr = ran_52 , marker='d', color='r', label='Value. '+today, uplims=True, lolims=True, fmt = '.')
    act_plot[-1][0].set_linestyle(':')
    act_plot[-1][1].set_linestyle(':')

    if 'abs' in typedif.lower() : profit = 'Prize [USD]'    
    else                        : profit = 'Profit/loss [%]'

    if 'bep' in typedif.lower() or '' in typedif.lower(): plt.scatter(stocknm, BEP    , marker='_',label='Break even point', color='blue', s=80)
    exp_plot = plt.errorbar(stocknm, exp_val, yerr =exp_ran, fmt='.', color='black', label='Exp. '+profit)
    exp_plot[-1][0].set_linestyle('--')
    plt.ylabel(profit)
    plt.title(typedif+' expected profit/loss')
    plt.legend()
    plt.tick_params(axis='x', rotation=90, labelsize=8)
    plt.grid( color='0.75', linestyle=':')
    plt.savefig(foredir+typedif.replace(' ','-').lower()+'_difference.png', dpi=300)
    plt.clf()

#Plot recommendation plots
def makerecoplots(sell, underw, hold, overw, buy, typedif):
    colrec  = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
    reconms = ['sell', 'underweight', 'hold', 'overwight', 'buy' ]
    order   = [4,3,2,1,0]
    plt.bar(stocknm,sell  , color=colrec[0], label=reconms[0])
    plt.bar(stocknm,underw, color=colrec[1], label=reconms[1], bottom=sell)
    plt.bar(stocknm,hold  , color=colrec[2], label=reconms[2], bottom=(sell+underw))
    plt.bar(stocknm,overw , color=colrec[3], label=reconms[3], bottom=sell+underw+hold)
    plt.bar(stocknm,buy   , color=colrec[4], label=reconms[4], bottom=sell+underw+hold+overw)

    perc = ''
    if 'rel' in typedif.lower() : perc+=' [%]'

    plt.title('Expert recommendations'+perc)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.ylabel('Exp. recom.'+perc)
    plt.tick_params(axis='x', rotation=90, labelsize=8)
    plt.savefig(foredir+typedif+'_recommendations.png', dpi=300)
    plt.clf()
    
def profitlossplots(profitloss, plot_type):
    if 'rel' in plot_type.lower():
        l_type  = '[%]'
        tit_ext = ' (average' + str(relprofitloss) + '%)'
    else:
        l_type  = '[EUR]'
        tit_ext = ' (total'+str(totprofitloss)+')'
    plt.axhline(0, color='black', linestyle='-')
    plt.bar(stocknm, profitloss, color= cprofitloss)
    plt.ylabel('Profit/loss '+l_type)
    plt.title(plot_type+' Profit/loss per stock'+tit_ext)
    plt.tick_params(axis='x', rotation=90, labelsize=8)
    plt.grid(axis='y', color='0.75', linestyle=':')
    plt.savefig(foredir+plot_type.lower()+'_profit-loss.png', dpi=300)
    plt.clf()

today    = str(datetime.date(datetime.now()))

if __name__ == '__main__':
    stocknm = []

    jsonDict  = open(act_info, 'rb')
    portfolio = json.load(jsonDict)

    stocknm    = np.array([])
    act_val    = np.array([])
    exp_val    = np.array([])
    exp_max    = np.array([])
    exp_min    = np.array([])
    BEP        = np.array([])
    profitloss = np.array([])
    size       = np.array([])
    tot_inv    = np.array([])

    max_1    = np.array([])
    max_52   = np.array([])
    min_1    = np.array([])
    min_52   = np.array([])

    buy     = np.array([])
    overw   = np.array([])
    hold    = np.array([])
    underw  = np.array([])
    sell    = np.array([])
    nrecos  = np.array([])
    reco_wi = np.array([])

    portfolio_or = (sorted(portfolio, key=lambda x: (portfolio[x][u'symbol'])))
    for stock_id in portfolio_or:
        if 'ETF' in portfolio[stock_id]['productType']:
            ETF_info = portfolio[stock_id]['productType']
            continue
        if '.D' in portfolio[stock_id][u'symbol']: continue
        if 'PRX' in portfolio[stock_id][u'symbol']: continue #quick fix
        stocknm  = np.append(stocknm , portfolio[stock_id][u'symbol'])
        print "stocknm", portfolio[stock_id][u'symbol']
        act_val    = np.append(act_val , portfolio[stock_id][u'act_val'])
        exp_val    = np.append(exp_val , portfolio[stock_id][u'exp_med'])
        exp_max    = np.append(exp_max , portfolio[stock_id][u'exp_max'])
        exp_min    = np.append(exp_min , portfolio[stock_id][u'exp_min'])
        BEP        = np.append(BEP     , portfolio[stock_id][u'BEP'])
        profitloss = np.append(profitloss, portfolio[stock_id][u'gain/loss'])
        size       = np.append(size    , portfolio[stock_id][u'size'])
        tot_inv    = np.append(tot_inv , portfolio[stock_id][u'BEP']*portfolio[stock_id][u'size'])

        max_1    = np.append(max_1   , portfolio[stock_id][u'max_1'])
        max_52   = np.append(max_52  , portfolio[stock_id][u'max_52'])
        min_1    = np.append(max_1   , portfolio[stock_id][u'min_1'])
        min_52   = np.append(min_52  , portfolio[stock_id][u'min_52'])

        recos   = portfolio[stock_id][u'recos']
        nrecos  = np.append(nrecos , portfolio[stock_id][u'nrecos'])
        reco_wi = np.append(reco_wi, portfolio[stock_id][u'reco_wi'])
        buy     = np.append(buy    , int( recos[0]))
        overw   = np.append(overw  , int( recos[1]))
        hold    = np.append(hold   , int( recos[2]))
        underw  = np.append(underw , int( recos[3]))
        sell    = np.append(sell   , int( recos[4]))
        
    #get differences
    exp_mindif      = exp_val-exp_min
    exp_maxdif      = exp_max-exp_val
    difference      = act_val-exp_val
    act_relval      = -100 + 100*act_val/act_val
    exp_relval      = -100 + 100*(exp_val/act_val)
    relBEP          = -100 + 100*BEP/BEP
    exp_relBEP      = -100 + 100*(exp_val/BEP)
    act_relBEP      = -100 + 100*(act_val/BEP)
    exp_minreldif   = 100*exp_mindif/act_val
    exp_maxreldif   = 100*exp_maxdif/act_val
    exp_relran      = np.array(list(zip(exp_minreldif, exp_maxreldif))).T
    exp_ran         = np.array(zip(exp_mindif, exp_maxdif)).T
    sum_totinv      = sum(tot_inv)
    totprofitloss   = round(sum(profitloss),3) 
    relprofitloss   = round(100*totprofitloss/sum_totinv)
    colors       = [ 'r' if i < 0 else 'g' for i in difference]
    cprofitloss  = [ 'r' if i < 0 else 'g' for i in profitloss]

    ran_52    = [act_val-min_52, max_52-act_val]
    relran_52 = 100*(ran_52/act_val)
    BEPran_52 = 100*(ran_52/BEP)

    #Profit/Loss plots
    profitlossplots(profitloss/(0.01*BEP*size), 'Relative')
    profitlossplots(profitloss                , 'Total')
    #Different pltos
    diff_plots(act_val   , exp_val   , BEP   ,ran_52   , 'Absolute')
    diff_plots(act_relval, exp_relval, BEP   ,relran_52, 'Relative')
    diff_plots(act_relBEP, exp_relBEP, relBEP,BEPran_52, 'relative BEP')
    #Recommendation plots
    makerecoplots(sell, underw, hold, overw, buy, 'Absolute')
    makerecoplots(100*sell/nrecos, 100* underw/nrecos, 100*hold/nrecos, 100*overw/nrecos, 100*buy/nrecos, 'Relative')

    plt.scatter(stocknm, reco_wi, c=reco_wi, cmap='RdYlGn_r')
    plt.clim(1,5)
    plt.grid(color='0.75', linestyle=':')
    plt.title('Buy/sell consensus (1-5) per stock')
    plt.tick_params(axis='x', rotation=90, labelsize=8)
    plt.savefig(foredir+'average_recommendation.png', dpi=300)

    
