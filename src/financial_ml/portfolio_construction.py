import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from financial_ml.common import DATA_DIR, PRED_FILE, SP500_MARKET_FILE

# Load Data for predictions, stocks, and SPY benchmark
preds = pd.read_csv(PRED_FILE)
preds['date'] = pd.to_datetime(preds['date'])
prices = pd.read_csv(SP500_MARKET_FILE, index_col=0, parse_dates=True).ffill()
spy = prices['SPY'] if 'SPY' in prices.columns else None
prices = prices.drop(columns=['SPY'], errors='ignore')

# Calculate monthly returns
returns_realized = prices.pct_change(periods=1).stack().reset_index()
returns_realized.columns = ['date', 'ticker', 'return']

# Merge Predictions with Returns
df = preds.merge(returns_realized, on=['date', 'ticker'], how='inner')

# Filter to best model (Logistic L2)
df = df[df['model'] == 'rf'].copy()

from financial_ml.portfolio_diagnostics import pre_filter_diagnostics
pre_filter_diagnostics(df)

# Construct Portfolio
def construct_portfolio(df, per_top=10, per_bot=10):
    """Long top per_top, short bottom per_bot
       keep ALL columns including date"""
    df['position'] = 0
    for date in df['date'].unique():
        mask = df['date'] == date
        date_data = df[mask].copy()
        top_subset = (100-per_top)/100.
        bot_subset = (per_bot/100.)
        top = date_data['y_prob'].quantile(top_subset)
        bottom = date_data['y_prob'].quantile(bot_subset)
        
        df.loc[mask & (df['y_prob'] >= top), 'position'] = 1
        df.loc[mask & (df['y_prob'] <= bottom), 'position'] = -1
    return df

df =construct_portfolio(df)

# Calculate Portfolio Returns
def calc_portfolio_return(group, type='100long'):
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

portfolio_returns = df.groupby('date').apply(calc_portfolio_return, include_groups=False).reset_index()
portfolio_returns.columns = ['date', 'portfolio_return']

# Cumulative returns
portfolio_returns['cum_return'] = (1 + portfolio_returns['portfolio_return']).cumprod()

# Benchmark (SPY buy-and-hold)
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

# Sharpe Ratio (annualized)
avg_return = portfolio_returns['portfolio_return'].mean() * 12  # Monthly to annual
volatility = portfolio_returns['portfolio_return'].std() * np.sqrt(12)
risk_free_rate = 0.04  # Assume 4% risk-free rate
sharpe_ratio = (avg_return - risk_free_rate) / volatility

# Maximum Drawdown
running_max = portfolio_returns['cum_return'].cummax()
drawdown = (portfolio_returns['cum_return'] - running_max) / running_max
max_drawdown = drawdown.min()

# Win Rate
win_rate = (portfolio_returns['portfolio_return'] > 0).sum() / len(portfolio_returns)

from financial_ml.portfolio_diagnostics import test_sharpe_significance

test_sharpe_significance(portfolio_returns=portfolio_returns, sharpe_ratio=sharpe_ratio)


print("=" * 60)
print("PORTFOLIO PERFORMANCE METRICS")
print("=" * 60)
print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
print(f"Max Drawdown:        {max_drawdown:.1%}")
print(f"Win Rate:            {win_rate:.1%}")
print(f"Annual Return:       {avg_return:.1%}")
print(f"Annual Volatility:   {volatility:.1%}")
print(f"Total Return:        {(portfolio_returns['cum_return'].iloc[-1] - 1):.1%}")
print("=" * 60)

# Visualizations
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Chart 1: Cumulative Returns
axes[0].plot(portfolio_returns['date'], portfolio_returns['cum_return'], 
             label='ML Strategy', linewidth=2, color='blue')
if spy is not None:
    axes[0].plot(portfolio_returns['date'], portfolio_returns['spy_cum_return'], 
                 label='SPY (Buy & Hold)', linewidth=2, color='gray', linestyle='--')
axes[0].set_title('Cumulative Returns: ML Strategy vs SPY', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cumulative Return (1 = start)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Chart 2: Drawdown Over Time
axes[1].fill_between(portfolio_returns['date'], drawdown * 100, 0, 
                      color='red', alpha=0.3, label='Drawdown')
axes[1].axhline(max_drawdown * 100, color='darkred', linestyle='--', 
                label=f'Max Drawdown: {max_drawdown:.1%}')
axes[1].set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Drawdown (%)', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/portfolio_backtest.png', dpi=300, bbox_inches='tight')
#plt.show()

print("\nChart saved to figures/portfolio_backtest.png")
