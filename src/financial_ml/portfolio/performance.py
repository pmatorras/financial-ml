"""
Calculate portfolio performance and metrics.
"""

import pandas as pd

def aggregate_portfolio_return(group, type='100long'):
    '''
    Calculate portfolio average return, for a single period by aggregatting positions.
    Args:
        group: 
        type: type of portfolio that one wants to follow
    Returns:
        df with the portfolio returns
    '''
    long_ret = group[group['position'] == 1]['return'].mean()
    short_ret = group[group['position'] == -1]['return'].mean()
    if type=='100long':
        portfolio_ret = long_ret if pd.notna(long_ret) else 0
    elif type =='longshort':
        # Dollar-neutral: 50% long, 50% short
        if pd.notna(long_ret) and pd.notna(short_ret):
            portfolio_ret = 0.5 * long_ret + 0.5 * (-short_ret)
        elif pd.notna(long_ret):
            portfolio_ret = 0.5 * long_ret  # Only long positions
        elif pd.notna(short_ret):
            portfolio_ret = 0.5 * (-short_ret)  # Only short positions
        else:
            portfolio_ret = 0    
    else:
        print(f"option {type} not implemented")
        exit()
    
    return portfolio_ret


def include_benchmark_return(spy, portfolio_returns):
    # Benchmark (SPY buy-and-hold)
    """
    Add SPY benchmark returns to portfolio DataFrame.
    
    Args:
        spy: Series with SPY prices (index=dates)
        portfolio_returns: DataFrame with portfolio returns and dates
        
    Returns:
        DataFrame with spy_return and spy_cum_return columns added
    """
    if spy is not None:
        spy_returns_raw = spy.pct_change().dropna()
        
        # Align by converting both to same format
        spy_df = pd.DataFrame({
            'date': spy_returns_raw.index,
            'spy_return': spy_returns_raw.values
        })
        
        # Merge with portfolio returns
        portfolio_returns = portfolio_returns.merge(spy_df, on='date', how='left')
        missing_spy = portfolio_returns['spy_return'].isna().sum()
        if missing_spy > 0:
            print(f"⚠️ WARNING: {missing_spy} months missing SPY data")
        portfolio_returns['spy_return'] = portfolio_returns['spy_return'].fillna(0)
        portfolio_returns['spy_cum_return'] = (1 + portfolio_returns['spy_return']).cumprod()
    else:
        print("WARNING, SPY is none!")
    return portfolio_returns