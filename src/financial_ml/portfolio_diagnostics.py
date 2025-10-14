import numpy as np
from scipy.stats import norm
import pandas as pd
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

