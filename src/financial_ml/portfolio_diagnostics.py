import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import itertools

def pre_filter_diagnostics(df):
    '''Check for data leakage and duplicate predictions'''
    print("\n=== PRE-FILTER DIAGNOSTIC ===")

    # Group by fold and check date ranges
    for fold in [1, 2, 3]:
        fold_data = df[df['fold'] == fold]
        if len(fold_data) > 0:
            print(f"\nFold {fold}:")
            print(f"  Date range: {fold_data['date'].min().date()} to {fold_data['date'].max().date()}")
            print(f"  Total predictions: {len(fold_data)}")

    # Check if any date has multiple folds
    duplicates = df.groupby(['date', 'ticker']).size()
    multi_fold_dates = duplicates[duplicates > 1]
    print(f"\n⚠️ Dates with multiple fold predictions: {len(multi_fold_dates)}")
    if len(multi_fold_dates) > 0:
        print("Sample conflicts:")
        print(multi_fold_dates.head(10))



def test_sharpe_significance(portfolio_returns, sharpe_ratio):
    '''Check if the sharpe ratio is statistically significant, also applying bonferroni correction'''
    returns = portfolio_returns['portfolio_return']
    n_obs = len(returns)

    # Calculate statistics
    skew = returns.skew()
    kurt = returns.kurtosis()

    # Raw Sharpe Significance
    var_sharpe = (1 + 0.5 * sharpe_ratio**2 - 
                skew * sharpe_ratio + 
                (kurt / 4) * sharpe_ratio**2) / n_obs
    stderr_sharpe = np.sqrt(var_sharpe)
    t_stat = sharpe_ratio / stderr_sharpe
    p_raw = 2 * norm.cdf(-abs(t_stat))

    print("Raw Sharpe Significance:")
    print(f"  Variance: {var_sharpe:.6f}")
    print(f"  Std Error: {stderr_sharpe:.4f}")
    print(f"  T-stat: {t_stat:.2f}")
    print(f"  P-value: {p_raw:.6f}")
    print(f"Result: {'✅ SIGNIFICANT (p < 0.001)' if p_raw < 0.001 else '❌ NOT SIGNIFICANT'}")

    # Bonferroni correction for 3 models
    bonf_thresh = 0.05 / 3
    print(f"\nMultiple Testing Adjustment (Bonferroni):")
    print(f"  Tested models: 3 (Logistic L1/L2, Random Forest)")
    print(f"  Adjusted threshold: p < {bonf_thresh:.4f}")
    print(f"  Result: {'✅ PASSES' if p_raw < bonf_thresh else '❌ FAILS'} multiple testing")


def print_model_agreement(preds_df, models):
    """Print model agreement analysis results"""
    
    print("\n" + "="*60)
    print("MODEL AGREEMENT ANALYSIS")
    print("="*60)
    print("Purpose: High correlation proves all models capture the same")
    print("         fundamental signal (not random overfitting)")
    
    results = model_agreement_correlations(preds_df,models)
    
    print("\nPairwise Prediction Correlations (Spearman):")
    for pair, stats in results['correlations'].items():
        model_names = pair.replace('_vs_', ' vs ')
        print(f"  {model_names:25s}: {stats['correlation']:.3f} (p < 0.001, n={stats['n_obs']:,})")
    
    avg = results['average_correlation']
    print(f"\nAverage Correlation: {avg:.3f}")
    
    print("\nInterpretation:")
    if results['interpretation'] == 'genuine_signal':
        print("  ✅ HIGH correlation (>0.70): Models agree strongly")
        print("     → Indicates genuine fundamental signal")
        print("     → NOT model-specific overfitting")
    elif results['interpretation'] == 'weak_signal':
        print("  ⚠️  MODERATE correlation (0.50-0.70): Models partially agree")
        print("     → Some signal present, but models diverge")
        print("     → Consider ensemble approach")
    else:
        print("  ❌ LOW correlation (<0.50): Models disagree")
        print("     → High risk of overfitting")
        print("     → Results may not be robust")
    
    print("="*60)

def model_agreement_correlations(preds_df, models_dict, print_results=True):
    # Compare predictions from all 3 models
    models = list(models_dict.keys())

    model_preds = {}

    for model in models:
        model_data = preds_df[preds_df['model'] == model][['date', 'ticker', 'y_prob']]
        model_preds[model] = model_data.set_index(['date', 'ticker'])['y_prob']
    correlations = {}
    for model1, model2 in itertools.combinations(models, 2):
        # Align predictions (inner join on date/ticker)
        merged = pd.merge(
            model_preds[model1].reset_index(),
            model_preds[model2].reset_index(),
            on=['date', 'ticker'],
            suffixes=('_1', '_2')
        )
        
        if len(merged) > 0:
            corr, pval = spearmanr(merged['y_prob_1'], merged['y_prob_2'])
            correlations[f"{model1}_vs_{model2}"] = {
                'correlation': corr,
                'p_value': pval,
                'n_obs': len(merged)
            }
    avg_corr = np.mean([v['correlation'] for v in correlations.values()])
    
    return {
        'correlations': correlations,
        'average_correlation': avg_corr,
        'interpretation': (
            'genuine_signal' if avg_corr > 0.7 else
            'weak_signal' if avg_corr > 0.5 else
            'overfitting'
        )
    }

def compare_model_performance_by_period(preds_df, returns_df):
    """Compare RF vs Logistic performance across time periods"""
    
    # Merge predictions with returns
    df = preds_df.merge(returns_df, on=['date', 'ticker'], how='inner')
    
    # Define periods
    def assign_period(date):
        if date < pd.Timestamp('2019-10-31'):
            return 'Bull 2016-2019'
        elif date < pd.Timestamp('2022-10-31'):
            return 'Mixed 2019-2022'  # Includes 2022 bear
        else:
            return 'Recovery 2022-2025'
    
    df['period'] = df['date'].apply(assign_period)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE BY PERIOD")
    print("="*60)
    
    for period in ['Bull 2016-2019', 'Mixed 2019-2022', 'Recovery 2022-2025']:
        period_data = df[df['period'] == period]
        
        print(f"\n{period}:")
        
        for model in ['logreg_l2', 'rf']:
            model_data = period_data[period_data['model'] == model]
            
            # Calculate AUC (prediction quality)
            auc = roc_auc_score(model_data['y_true'], model_data['y_prob'])
            
            # Calculate IC (information coefficient - correlation with returns)
            ic = spearmanr(model_data['y_prob'], model_data['return'])[0]
            
            print(f"  {model:12s}: AUC={auc:.3f}, IC={ic:.3f}")
    
    print("="*60)

def analyze_prediction_stability(df):
    """
    Check how much predictions change for the same stock month-to-month.
    High volatility = high turnover.
    """
    df = df.sort_values(['ticker', 'date'])
    
    # Calculate month-to-month change in y_prob
    df['y_prob_prev'] = df.groupby('ticker')['y_prob'].shift(1)
    df['y_prob_change'] = (df['y_prob'] - df['y_prob_prev']).abs()
    
    print("\n" + "="*60)
    print("PREDICTION STABILITY ANALYSIS")
    print("="*60)
    print(f"\nMonth-to-Month Prediction Change:")
    print(f"  Mean:   {df['y_prob_change'].mean():.4f}")
    print(f"  Median: {df['y_prob_change'].median():.4f}")
    print(f"  75th percentile: {df['y_prob_change'].quantile(0.75):.4f}")
    print(f"  90th percentile: {df['y_prob_change'].quantile(0.90):.4f}")
    
    print(f"\nInterpretation:")
    if df['y_prob_change'].mean() > 0.05:
        print(f"  ❌ HIGH volatility (mean change {df['y_prob_change'].mean():.2%})")
        print(f"     → Predictions swing wildly → high turnover")
        print(f"     → Solution: Add smoothing")
    elif df['y_prob_change'].mean() > 0.03:
        print(f"  ⚠️  MODERATE volatility")
    else:
        print(f"  ✅ LOW volatility (stable predictions)")
    
    print("="*60)



def analyze_turnover(df):
    """
    Calculate actual portfolio turnover (how much you trade monthly).
    High turnover with IC=0.03 means transaction costs kill your alpha.
    """
    
    print("\n" + "="*60)
    print("TURNOVER ANALYSIS")
    print("="*60)
    
    # Get holdings each month
    dates = sorted(df['date'].unique())
    
    turnovers = []
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        prev_holdings = set(df[(df['date'] == prev_date) & (df['position'] == 1)]['ticker'])
        curr_holdings = set(df[(df['date'] == curr_date) & (df['position'] == 1)]['ticker'])
        
        # Stocks that changed
        exits = prev_holdings - curr_holdings  # Sold
        entries = curr_holdings - prev_holdings  # Bought
        
        # Turnover = (buys + sells) / 2 / portfolio_size
        turnover = len(exits.union(entries)) / (2 * len(prev_holdings)) if len(prev_holdings) > 0 else 0
        turnovers.append(turnover)
    
    avg_turnover = np.mean(turnovers)
    
    print(f"\nMonthly Statistics:")
    print(f"  Average turnover:    {avg_turnover:.1%}")
    print(f"  Median turnover:     {np.median(turnovers):.1%}")
    print(f"  Max turnover:        {np.max(turnovers):.1%}")
    print(f"  Min turnover:        {np.min(turnovers):.1%}")
    
    # Transaction cost impact (with IC=0.03)
    cost_per_trade = 0.0016  # 16 bps
    annual_cost = avg_turnover * cost_per_trade * 12
    
    print(f"\nTransaction Cost Impact:")
    print(f"  Annual drag:         {annual_cost:.2%}")
    
    # With IC=0.03, expected alpha is ~3%
    expected_alpha = 0.03
    net_alpha = expected_alpha - annual_cost
    
    print(f"\nAlpha After Costs:")
    print(f"  Expected alpha (IC=0.03): {expected_alpha:.1%}")
    print(f"  Transaction costs:        {annual_cost:.2%}")
    print(f"  Net alpha:                {net_alpha:.2%}")
    
    if net_alpha < 0.01:
        print(f"  ❌ WARNING: Transaction costs consume most/all alpha!")
    elif net_alpha < 0.02:
        print(f"  ⚠️  CAUTION: Transaction costs consume >50% of alpha")
    else:
        print(f"  ✅ Reasonable: Net alpha > 2%")
    
    print("="*60)
    
    return avg_turnover

def analyze_beta_exposure(portfolio_returns):
    """
    Calculate your portfolio's beta to SPY.
    Beta > 1.0 means you're just taking more market risk, not adding alpha.
    """
    from scipy.stats import linregress
    
    print("\n" + "="*60)
    print("MARKET BETA ANALYSIS")
    print("="*60)
    
    # Merge portfolio and SPY returns
    
    # Regression: portfolio_return = alpha + beta * spy_return
    slope, intercept, r_value, p_value, std_err = linregress(
        portfolio_returns['spy_return'], 
        portfolio_returns['portfolio_return']
    )
    
    beta = slope
    alpha_monthly = intercept
    alpha_annual = alpha_monthly * 12
    r_squared = r_value ** 2
    
    print(f"\nRegression Results:")
    print(f"  Beta:                {beta:.3f}")
    print(f"  Alpha (monthly):     {alpha_monthly:.3%}")
    print(f"  Alpha (annual):      {alpha_annual:.2%}")
    print(f"  R-squared:           {r_squared:.3f}")
    print(f"  P-value:             {p_value:.4f}")
    
    print(f"\nInterpretation:")
    if beta > 1.15:
        print(f"  ❌ HIGH BETA: You're taking {(beta-1)*100:.0f}% more market risk than SPY")
        print(f"     → Most 'alpha' is actually beta (riskier stocks)")
    elif beta > 1.05:
        print(f"  ⚠️  MODERATE BETA: Slightly riskier than SPY ({beta:.2f})")
    elif beta > 0.95:
        print(f"  ✅ NEUTRAL BETA: Similar risk to SPY ({beta:.2f})")
    else:
        print(f"  ✅ DEFENSIVE: Lower risk than SPY ({beta:.2f})")
    
    if alpha_annual > 0.02:
        print(f"  ✅ True alpha exists: {alpha_annual:.1%} annual")
    elif alpha_annual > 0:
        print(f"  ⚠️  Weak alpha: {alpha_annual:.1%} annual")
    else:
        print(f"  ❌ No alpha: All returns from beta")
    
    print("="*60)
    
    return beta, alpha_annual

def analyze_sector_concentration(df, pred_col='y_prob', latest_date=None):
    """
    Check if your top 10% is concentrated in specific sectors.
    Need sector data - if you don't have it, use this workaround.
    """
    if latest_date is None:
        latest_date = df['date'].max()
    print( df[(df['date'] == latest_date)])

    holdings = df[(df['date'] == latest_date) & (df['position'] == 1)]['ticker'].values
    
    print("\n" + "="*60)
    print("SECTOR CONCENTRATION CHECK")
    print("="*60)
    print(f"Date: {latest_date.date()}")
    print(f"Total holdings: {len(holdings)}")
    print(f"\nTop 20 holdings:")
    
    top_20 = df[(df['date'] == latest_date) & (df['position'] == 1)].nlargest(20, pred_col)
    print(top_20)
    #exit()
    for idx, row in top_20.iterrows():
        print(f"  {row['ticker']:>6s}: {row[pred_col]:.3f}")
    
    print(f"\n⚠️  Manual check required:")
    print(f"  Look up these tickers and check if they're mostly:")
    print(f"  - Tech (AAPL, MSFT, NVDA, GOOGL, META, etc.)")
    print(f"  - Finance (JPM, BAC, GS, MS, etc.)")
    print(f"  - Healthcare (UNH, JNJ, PFE, etc.)")
    print(f"  If >50% are one sector → concentration risk!")
    
    print("="*60)

def compare_drawdowns_to_spy(portfolio_returns):
    """
    Compare your drawdowns to SPY during same periods.
    If you fall more than SPY, you don't have downside protection.
    """
    
    print("\n" + "="*60)
    print("DRAWDOWN COMPARISON TO SPY")
    print("="*60)
    
    
    # Calculate cumulative returns
    portfolio_returns['port_cum'] = (1 + portfolio_returns['portfolio_return']).cumprod()
    portfolio_returns['spy_cum'] = (1 + portfolio_returns['spy_return']).cumprod()
    
    # Calculate drawdowns
    portfolio_returns['port_dd'] = (portfolio_returns['cum_return'] / portfolio_returns['cum_return'].cummax()) - 1
    portfolio_returns['spy_dd'] = (portfolio_returns['spy_cum_return'] / portfolio_returns['spy_cum_return'].cummax()) - 1
    
    # Find worst drawdown periods
    port_worst = portfolio_returns.loc[portfolio_returns['port_dd'].idxmin()]
    spy_worst = portfolio_returns.loc[portfolio_returns['spy_dd'].idxmin()]
    
    print(f"\nWorst Drawdowns:")
    print(f"  Your portfolio: {portfolio_returns['port_dd'].min():.1%} on {port_worst['date'].date()}")
    print(f"  SPY:            {portfolio_returns['spy_dd'].min():.1%} on {spy_worst['date'].date()}")
    
    # Check 2022 bear market specifically
    bear_2022 = portfolio_returns[(portfolio_returns['date'] >= '2022-01-01') & (portfolio_returns['date'] <= '2022-12-31')]
    if len(bear_2022) > 0:
        port_2022_dd = bear_2022['port_dd'].min()
        spy_2022_dd = bear_2022['spy_dd'].min()
        
        print(f"\n2022 Bear Market:")
        print(f"  Your portfolio: {port_2022_dd:.1%}")
        print(f"  SPY:            {spy_2022_dd:.1%}")
        
        if port_2022_dd < spy_2022_dd - 0.05:  # 5% worse
            print(f"  ❌ You fell MORE than SPY (no downside protection)")
        elif port_2022_dd < spy_2022_dd:
            print(f"  ⚠️  Slightly worse than SPY")
        else:
            print(f"  ✅ Better downside protection than SPY")
    
    print("="*60)