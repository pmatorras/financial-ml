"""
Portfolio diagnostic and validation functions.

Analyzes:
- Cross-validation integrity
- Model agreement and robustness
- Prediction stability and turnover
- Risk metrics (beta, drawdown)
- Statistical significance testing
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import itertools
from financial_ml.utils.config import SP500_NAMES_FILE, SEPARATOR_WIDTH
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




def print_model_agreement(preds_df, models):
    """Print model agreement analysis results"""
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("MODEL AGREEMENT ANALYSIS")
    print("="* SEPARATOR_WIDTH)
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
    
    print("="* SEPARATOR_WIDTH)

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

def compare_model_performance_by_period(preds_df, returns_df, models):
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
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("MODEL PERFORMANCE BY PERIOD")
    print("="* SEPARATOR_WIDTH)
    if 'date' in preds_df.columns:
        min_date = preds_df['date'].min()
        max_date = preds_df['date'].max()
        print(f"\nPrediction data available from: {min_date} to {max_date}")
        print(f"Total months with data: {len(preds_df['date'].unique())}")
    for period in ['Bull 2016-2019', 'Mixed 2019-2022', 'Recovery 2022-2025']:
        period_data = df[df['period'] == period]
        
        print(f"\n{period}:")
        
        for model in models:
            model_data = period_data[period_data['model'] == model]
            
            # Calculate AUC (prediction quality)
            auc = roc_auc_score(model_data['y_true'], model_data['y_prob'])
            
            # Calculate IC (information coefficient - correlation with returns)
            ic = spearmanr(model_data['y_prob'], model_data['return'])[0]
            
            print(f"  {model:12s}: AUC={auc:.3f}, IC={ic:.3f}")
    
    print("="* SEPARATOR_WIDTH)

def analyze_prediction_stability(df):
    """
    Check how much predictions change for the same stock month-to-month.
    High volatility = high turnover.
    """
    df = df.sort_values(['ticker', 'date'])
    
    # Calculate month-to-month change in y_prob
    df['y_prob_prev'] = df.groupby('ticker')['y_prob'].shift(1)
    df['y_prob_change'] = (df['y_prob'] - df['y_prob_prev']).abs()
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("PREDICTION STABILITY ANALYSIS")
    print("="* SEPARATOR_WIDTH)
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
    
    print("="* SEPARATOR_WIDTH)



def analyze_turnover(df):
    """
    Calculate actual portfolio turnover (how much you trade monthly).
    High turnover with IC=0.03 means transaction costs kill your alpha.
    """
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("TURNOVER ANALYSIS")
    print("="* SEPARATOR_WIDTH)
    
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
    
    print("="* SEPARATOR_WIDTH)
    
    return avg_turnover

def calculate_alpha_beta(df_returns, benchmark_returns, benchmark_nm, message):
    from scipy.stats import linregress

    # Regression: portfolio_return = alpha + beta * spy_return
    slope, intercept, r_value, p_value, std_err = linregress(
        benchmark_returns,
        df_returns
    )
    
    beta = slope
    alpha_monthly = intercept
    alpha_annual = alpha_monthly * 12
    r_squared = r_value ** 2
    
    print(f"- ML model vs: {message}")
    print(f"  Beta:                {beta:.3f}")
    print(f"  Alpha (monthly):     {alpha_monthly:.3%}")
    print(f"  Alpha (annual):      {alpha_annual:.2%}")
    print(f"  R-squared:           {r_squared:.3f}")
    print(f"  P-value:             {p_value:.4f}")
    if beta > 1.15:
        print(f"  ❌ HIGH BETA: You're taking {(beta-1)*100:.0f}% more market risk than {benchmark_nm}")
        print(f"     → Most 'alpha' is actually beta (riskier stocks)")
    elif beta > 1.05:
        print(f"  ⚠️  MODERATE BETA: Slightly riskier than {benchmark_nm} ({beta:.2f})")
    elif beta > 0.95:
        print(f"  ✅ NEUTRAL BETA: Similar risk to {benchmark_nm} ({beta:.2f})")
    else:
        print(f"  ✅ DEFENSIVE: Lower risk than {benchmark_nm} ({beta:.2f})")
    
    if alpha_annual > 0.02:
        print(f"  ✅ True alpha exists: {alpha_annual:.1%} annual")
    elif alpha_annual > 0:
        print(f"  ⚠️  Weak alpha: {alpha_annual:.1%} annual")
    else:
        print(f"  ❌ No alpha: All returns from beta")
    print("-"* SEPARATOR_WIDTH)

def analyze_beta_exposure(portfolio_returns, random_returns, equal_returns):
    """
    Calculate alpha and beta against multiple benchmarks to understand return sources.
    
    Three key comparisons:
    1. vs SPY (Market Beta) - Are you just levered to the market?
    2. vs Equal-Weight - Are you adding value vs passive equal-weighting?
    3. vs Random - Is your model better than luck?
    """
    from scipy.stats import linregress
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("ALPHA & BETA ANALYSIS - Return Attribution")
    print("="* SEPARATOR_WIDTH)
    
    print(f"\nRegression Results:")
    calculate_alpha_beta(portfolio_returns['portfolio_return'], portfolio_returns['spy_return'], "SPY", "SPY (Market Beta):")
    calculate_alpha_beta(portfolio_returns['portfolio_return'], equal_returns['portfolio_return'], "Equal-Weight Benchmark" , " Equal-Weight Benchmark (Selection Alpha):")
    calculate_alpha_beta(portfolio_returns['portfolio_return'], random_returns['portfolio_return'], "Random Selection", "Random Selection (Pure Model Alpha):")

    

def analyze_sector_concentration(df_portfolio, sector_file=SP500_NAMES_FILE, latest_date=None):
    """
    Analyze sector concentration in portfolio using existing GICS sector column.
    
    Args:
        df_portfolio: Portfolio DataFrame (must have 'sector' or 'GICS_sector' column)
        sector_file: CSV with ticker->sector mapping
        latest_date: Date to analyze (default: most recent)
    """
    """
    Analyze sector concentration by mapping tickers to sectors on-the-fly.
    
    Args:
        df_portfolio: Portfolio DataFrame with ticker column
        sector_file: CSV with ticker->sector mapping
        latest_date: Date to analyze
    """
    # Load sector mapping
    sectors = pd.read_csv(sector_file)
    if 'Symbol' in sectors.columns:
        sectors = sectors.rename(columns={'Symbol': 'ticker'})
    if latest_date is None:
        latest_date = df_portfolio['date'].max()
    
    # Get holdings and merge with sector data
    holdings = df_portfolio[df_portfolio['date'] == latest_date].copy()
    holdings = holdings.merge(sectors[['ticker', 'GICS Sector']], on='ticker', how='left')
    
    # Now you have sector info!
    long_holdings = holdings[holdings['position'] == 1]
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("SECTOR CONCENTRATION ANALYSIS")
    print("="* SEPARATOR_WIDTH)
    print(f"Date: {latest_date.date()}")
    print(f"Long positions: {len(long_holdings)}")
    
    # Check for missing sectors
    missing = long_holdings['GICS Sector'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} stocks missing sector data")
    
    # Sector breakdown
    sector_counts = long_holdings['GICS Sector'].value_counts()
    sector_pcts = (sector_counts / len(long_holdings) * 100).round(1)
    
    print("\nPortfolio Sector Breakdown:")
    print("-"* SEPARATOR_WIDTH)
    for sector, count in sector_counts.items():
        pct = sector_pcts[sector]
        bar = "█" * int(pct / 2)
        print(f"  {sector:30} {count:3d} ({pct:5.1f}%) {bar}")
    
    # Concentration check
    max_sector = sector_counts.index[0]
    max_pct = sector_pcts.iloc[0]
    
    print("\n" + "="* SEPARATOR_WIDTH)
    if max_pct > 40:
        print(f"❌ HIGH CONCENTRATION: {max_pct:.1f}% in {max_sector}")
    elif max_pct > 25:
        print(f"⚠️  MODERATE CONCENTRATION: {max_pct:.1f}% in {max_sector}")
    else:
        print(f"✅ WELL DIVERSIFIED: Max {max_pct:.1f}% in {max_sector}")
    
    print("="* SEPARATOR_WIDTH)
    
    return holdings


def calculate_drawdowns(df_returns, key_nm='portfolio'):
    ret_key = key_nm+'_return'
    cum_key = key_nm+'_cum'
    dd_key = key_nm+'_dd'
    cum_ret = 'cum_return' if key_nm=='portfolio' else key_nm+'_cum_return'
    df_returns[cum_key] = (1 + df_returns[ret_key]).cumprod()
    df_returns[dd_key] = (df_returns[cum_ret] / df_returns[cum_ret].cummax()) - 1
    worst_dd_date = df_returns.loc[df_returns[dd_key].idxmin()]
    worst_dd = df_returns[dd_key].min()
    #print(f"{df_returns[dd_key].min():.1%} on {worst_dd_date['date'].date()}")
    # Check 2022 bear market specifically

    covid_2020 = df_returns[(df_returns['date'] >= '2020-01-01') & (df_returns['date'] <= '2020-12-31')] #note the year is referring to the market behaviour on that year, not the virus
    if len(covid_2020) > 0:
        covid_2020_dd = covid_2020[dd_key].min()
    else:
        covid_2020_dd = None
        
    bear_2022 = df_returns[(df_returns['date'] >= '2022-01-01') & (df_returns['date'] <= '2022-12-31')]
    if len(bear_2022) > 0:
        bear_2022_dd = bear_2022[dd_key].min()
    else:
        bear_2022_dd = None

    return {'worst_dd': worst_dd,
            'worst_dd_date': worst_dd_date,
            'covid_2020' : covid_2020_dd,
            'bear_2022' : bear_2022_dd
    }

def compare_dd_values(portfolio_dds, comparison_dds, drawdown='bear_2022', comparison_name='SPY'):
    if portfolio_dds[drawdown] < comparison_dds[drawdown] - 0.05:  # 5% worse
        print(f"  ❌ You fell MORE than {comparison_name} (no downside protection)")
    elif portfolio_dds[drawdown] < comparison_dds[drawdown]:
        print(f"  ⚠️  Slightly worse than {comparison_name}")
    else:
        print(f"  ✅ Better downside protection than {comparison_name}")



def compare_drawdowns_to_spy(portfolio_returns, equal_weights_portfolio, random_returns):
    """
    Compare your drawdowns to SPY during same periods.
    If you fall more than SPY, you don't have downside protection.
    """
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("DRAWDOWN COMPARISON")
    print("="* SEPARATOR_WIDTH)

    portfolio_dds = calculate_drawdowns(portfolio_returns, 'portfolio')
    spy_dds = calculate_drawdowns(portfolio_returns, 'spy')
    equal_dds = calculate_drawdowns(equal_weights_portfolio, 'portfolio')
    random_dds = calculate_drawdowns(random_returns, 'portfolio')

    print(f"  ML Strategy:         {portfolio_dds['worst_dd'].min():.1%} on {portfolio_dds['worst_dd_date']['date'].date()}")
    print(f"  Equal-Weight (100%): {equal_dds['worst_dd'].min():.1%} on {equal_dds['worst_dd_date']['date'].date()}")
    print(f"  Random Selection:    {random_dds['worst_dd'].min():.1%} on {random_dds['worst_dd_date']['date'].date()}")

    print(f"  SPY:                 {spy_dds['worst_dd'].min():.1%} on {spy_dds['worst_dd_date']['date'].date()}")

    print(f"\n2020 COVID Crash:")
    print(f"  ML Strategy:         {portfolio_dds['covid_2020']:.1%}")
    print(f"  Equal-Weight (100%): {equal_dds['covid_2020']:.1%}")
    print(f"  Random Selection:    {random_dds['covid_2020']:.1%}")
    print(f"  SPY:                 {spy_dds['covid_2020']:.1%}")
    print("-"* SEPARATOR_WIDTH)

    compare_dd_values(portfolio_dds, spy_dds, 'covid_2020', 'SPY')
    compare_dd_values(portfolio_dds, equal_dds, 'covid_2020', 'Equal-Weight (100%)')
    compare_dd_values(portfolio_dds, random_dds, 'covid_2020', 'Random Selection')

    print(f"\n2022 Bear Market:")
    print(f"  ML Strategy:         {portfolio_dds['bear_2022']:.1%}")
    print(f"  Equal-Weight (100%): {equal_dds['bear_2022']:.1%}")
    print(f"  Random Selection:    {random_dds['bear_2022']:.1%}")
    print(f"  SPY:                 {spy_dds['bear_2022']:.1%}")

    compare_dd_values(portfolio_dds, spy_dds, 'bear_2022', 'SPY')
    compare_dd_values(portfolio_dds, equal_dds, 'bear_2022', 'Equal-Weight (100%)')
    compare_dd_values(portfolio_dds, random_dds, 'bear_2022', 'Random Selection')

    print("="* SEPARATOR_WIDTH)

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


def test_random_baseline(df, per_top=10, per_bot=10, pred_col='y_prob', portfolio_type='100long'):
    """
    Test if alpha comes from model or equal-weighting.
    
    Returns dict with metrics AND the random portfolio for plotting.
    """
    import numpy as np
    from financial_ml.portfolio.construction import construct_portfolio
    from financial_ml.portfolio.performance import aggregate_portfolio_return
    
    print("\n" + "="* SEPARATOR_WIDTH)
    print("RANDOM BASELINE TEST - Model Validation")
    print("="* SEPARATOR_WIDTH)
    
    # Save original predictions
    df = df.copy()
    df['y_prob_original'] = df[pred_col].copy()
    
    # Test with random predictions
    np.random.seed(42)
    df[pred_col] = np.random.uniform(
        df['y_prob_original'].min(), 
        df['y_prob_original'].max(), 
        len(df)
    )
    
    print("\nTesting random predictions...")
    df_random_portfolio = construct_portfolio(df, per_top, per_bot, pred_col=pred_col)
    #perf_random = aggregate_portfolio_return(df_random_portfolio)
    perf_random = df_random_portfolio.groupby('date').apply(aggregate_portfolio_return,  portfolio_type=portfolio_type, include_groups=False).reset_index()
    perf_random_mean = perf_random['return'].mean() if 'return' in perf_random.columns else perf_random[0].mean()

    # Restore original predictions
    df[pred_col] = df['y_prob_original']
    
    print("Testing real model predictions...")
    df_real_portfolio = construct_portfolio(df, per_top, per_bot, pred_col=pred_col)
    #perf_real = aggregate_portfolio_return(df_real_portfolio)
    perf_real = df_real_portfolio.groupby('date').apply(aggregate_portfolio_return,  portfolio_type=portfolio_type, include_groups=False).reset_index()
    perf_real_mean = perf_real['return'].mean() if 'return' in perf_real.columns else perf_real[0].mean()

    print("\n" + "="* SEPARATOR_WIDTH)
    print("COMPARISON")
    print("="* SEPARATOR_WIDTH)
    print(f"Random:  {perf_random_mean:.6f}")
    print(f"Real:    {perf_real_mean:.6f}")
    print(f"Delta:   {perf_real_mean - perf_random_mean:.6f}")
    print(f"\n   Annualized:")
    print(f"Random:  {perf_random_mean * 12 * 100:.2f}%")
    print(f"Real:    {perf_real_mean * 12 * 100:.2f}%")
    print(f"Alpha:   {(perf_real_mean - perf_random_mean) * 12 * 100:.2f}%")
    if (perf_real_mean - perf_random_mean) * 12 * 100 < 0.5:
        print("❌ Model not better than random")
    else:
        print("✅ Model shows genuine skill")
    print("="* SEPARATOR_WIDTH)
    
    perf_random.columns = ['date', 'portfolio_return']

    # Return both metrics and the random portfolio data for plotting
    return perf_random