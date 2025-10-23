"""
Engineer the required features for the machine learning
"""

import warnings
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BQuarterEnd

from financial_ml.utils.helpers import safe_div
from financial_ml.utils.config import DEBUG_DIR
from financial_ml.data.validation import require_non_empty


def to_monthly_ffill(df, maxdate=None, freq="BM", lag_months=1):
    """
    Minimal monthly duplication by period_end:
    - Groups by (ticker, metric, unit, canonical_key)
    - Resamples at monthly frequency on period_end
    - Forward-fills to duplicate values between quarter ends
    - ADDS LAG to simulate reporting delay
    """
    gcols = ["ticker", "metric", "unit", "canonical_key"]

    #remove duplicates, keep the last filing
    df_clean = (
        df.copy()
        .assign(
            period_end=pd.to_datetime(df["period_end"], errors="coerce", utc=True).dt.tz_convert(None),
            filed=pd.to_datetime(df["filed"], errors="coerce", utc=True).dt.tz_convert(None)
        )
        .sort_values(gcols + ["filed", "period_end"])
        .drop_duplicates(subset=gcols + ["filed"], keep="last")
    )
    # Add configurable lag
    if lag_months > 0:
        df_clean = df_clean.assign(
            filed=lambda x: x['filed'] + pd.DateOffset(months=lag_months)
        )
    out = (
        df_clean.copy()
          .assign(filed=pd.to_datetime(df["filed"], errors="coerce", utc=True).dt.tz_convert(None))
          .sort_values(gcols + ["filed"])
          .set_index("filed")
          .groupby(gcols, dropna=False)["value"]
          .resample(freq)
          .ffill()
          .rename("value")
          .reset_index()
          .rename(columns={"filed": "period_end"})  
    )
    out = (out.sort_values(["ticker", "canonical_key", "period_end"])
              .drop_duplicates(subset=["ticker", "canonical_key", "period_end"], keep="last")
              .reset_index(drop=True))
    if maxdate is not None:
        maxdate = pd.to_datetime(maxdate)
        out = out[out["period_end"] <= maxdate]

    return out.drop_duplicates(subset=gcols + ['period_end'], keep='last')



def widen_by_canonical(monthly, prices, canonical_key):
    '''
    Convert long-format fundamentals into a wide format that is align with the prices.
    Args:
        monthly: Long-format DataFrame with canonical_key column
        prices: Price DataFrame for alignment
        canonical_key: Which concept to extract (e.g., "Assets")
    '''
    monthly["period_end"] = pd.to_datetime(monthly["period_end"], errors="coerce")
    wide = (monthly.loc[monthly["canonical_key"] == canonical_key, ["period_end","ticker","value"]]
                   .pivot(index="period_end", columns="ticker", values="value")
                   .sort_index()
                   .reindex(index=prices.index, columns=prices.columns))
    require_non_empty(wide, f"wide_{canonical_key}")
    return wide


#MARKET FEATURES
def calculate_market_features(prices,args):
    """
    Calculate technical market features from prices.
    
    Args:
        prices: DataFrame with stock prices (wide format: dates × tickers)
        
    Returns:
        dict: {feature_name: DataFrame} for each market feature
    """
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    ret_1m = prices.pct_change(1, fill_method=None) #1 month change
    ret_12m = prices.pct_change(12, fill_method=None) #change in 12 months
    mom_12_1 = (1 + ret_12m) / (1 + ret_1m) - 1 #month on month momentum
    vol_3m = ret_1m.rolling(3).std() #volumes
    vol_12m = ret_1m.rolling(12).std()

    if args.debug: ret_1m.to_csv(DEBUG_DIR/"testret.csv")
    return {
        "ClosePrice": prices,
        "r1": ret_1m,
        "r12": ret_12m,
        "mom121": mom_12_1,
        "vol3": vol_3m,
        "vol12": vol_12m
    }

def compute_log_mktcap(price: pd.Series, shares: pd.Series, name="LogMktCap") -> pd.Series:
    '''
    Function to deal with sometimes empty mkcap values.
    It calculates it when possible, if not raises a warning instead of dealing with infinities
    '''
    # Ensure numeric
    p = pd.to_numeric(price, errors="coerce")
    s = pd.to_numeric(shares, errors="coerce")

    # Market cap
    mcap = (p * s).astype(float)

    # Identify problematic inputs for log
    bad = ~np.isfinite(mcap) | (mcap <= 0)

    # Raise a single warning with counts/ratio if any bad values exist
    if bad.any():
        n_bad = int(bad.sum())
        frac = float(n_bad) / float(mcap.size)
        warnings.warn(
            f"LogMktCap: {n_bad} ({frac:.2%}) nonpositive/missing market cap values; "
            "these entries will be left as NaN.",
            category=UserWarning
        )

    # Compute log only on valid entries; leave others as NaN
    out = pd.Series(np.nan, index=mcap.index, name=name)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(mcap) & (mcap > 0)
        out.loc[valid] = np.log(mcap.loc[valid].to_numpy())

    return out



def compute_fundamental_ratios(feat_long, args):
    """
    Calculate financial ratios from fundamental data, input as a long df.
    
    Args:
        feat_long: Long-format DataFrame with base fundamentals
        debug: Save intermediate outputs
        
    Returns:
        feat_long with added ratio columns
    """
    if args.debug: feat_long.to_csv(DEBUG_DIR/"feat_long.csv")

    #Extract base metrics
    price = feat_long["ClosePrice"].astype(float)
    n_shares = feat_long["CommonStockSharesOutstanding"].astype(float)
    assets = feat_long["Assets"].astype(float)
    liabilities = feat_long["Liabilities"].astype(float)
    equity = feat_long["StockholdersEquity"].astype(float)
    net_income = feat_long["NetIncomeLoss"]
    revenues = feat_long["Revenues"]

    #Market cap and size
    market_cap = price * n_shares
    mcap = market_cap.astype(float)
    feat_long['LogMktCap'] = compute_log_mktcap( price, n_shares, name='LogMktCap')
    feat_long["MarketEquity"] = market_cap

    if args.debug:
        bad = ~np.isfinite(mcap) | (mcap <= 0)
        n_shares.to_csv(DEBUG_DIR/"nshares.csv") 
        revenues.to_csv(DEBUG_DIR/'revenues.csv')
        print("nonpositive_or_missing:", bad.sum())           # count of bad inputs
        print(mcap[bad].head())                                # spot-check values
        mcap.to_csv(DEBUG_DIR/"marketcap.csv")
        feat_long['LogMktCap'].to_csv(DEBUG_DIR/'mkap.csv')

    #Valuation
    feat_long["BookToMarket"] = safe_div(equity, market_cap)

    # Profitability
    equity_prev = equity.groupby(level=1).shift(1)
    avg_equity = (equity + equity_prev) / 2.0
    feat_long["ROE"] = safe_div(net_income, avg_equity)
    feat_long["ROA"] = safe_div(net_income, assets)
    feat_long["NetMargin"] = safe_div(net_income, revenues)

    # Leverage
    feat_long["Leverage"] = safe_div(liabilities, assets)

    # Cuarterly growth metrics
    dates = pd.Series(feat_long.index.get_level_values('Date'))
    is_bqe = (dates == pd.DatetimeIndex([BQuarterEnd().rollback(d) for d in dates])).values
    bq_roll = pd.DatetimeIndex([BQuarterEnd().rollback(d) for d in dates])

    assets_qe = assets[is_bqe]
    shares_qe = n_shares[is_bqe]
    assets_lag4 = assets_qe.groupby(level=1).shift(4)
    shares_lag4 = shares_qe.groupby(level=1).shift(4)

    inv_qe = (assets_qe - assets_lag4) / assets_lag4
    iss_qe = (shares_qe - shares_lag4) / shares_lag4
    inv_qe = inv_qe.replace([np.inf, -np.inf], np.nan)
    iss_qe = iss_qe.replace([np.inf, -np.inf], np.nan)
    if args.debug:
        print(feat_long.keys(), feat_long["ROE"])
        check = pd.DataFrame({'Date': dates, 'BQ_Rollback': bq_roll, 'Is_BQ_End': is_bqe, }).set_index('Date')
        check.to_csv(DEBUG_DIR/"check_dates.csv", mode='w')

    # Write quarterly results back and forward-fill to months for the asset growth and netshare insurance"    
    feat_long['AssetGrowth'] = np.nan
    feat_long.loc[inv_qe.index, 'AssetGrowth'] = inv_qe
    feat_long['AssetGrowth'] = feat_long['AssetGrowth'].groupby(level=1).ffill()

    feat_long['NetShareIssuance'] = np.nan
    feat_long.loc[iss_qe.index, 'NetShareIssuance'] = iss_qe
    feat_long['NetShareIssuance'] = feat_long['NetShareIssuance'].groupby(level=1).ffill()

    return feat_long





def create_binary_labels(prices, spy_benchmark):
    """
    Create binary classification labels for outperformance vs SPY.
    
    Args:
        prices: Stock prices
        spy_benchmark: SPY benchmark Series
        
    Returns:
        Series with binary labels (1 = outperform, 0 = underperform)
    """
        # Label: 12m forward total return > 0
    stock_fwd12 = (prices.shift(-12) / prices) - 1  # (future / now) - 1
    spy_benchmark_fwd12 = (spy_benchmark.shift(-12) / spy_benchmark) - 1   
    excess_fwd12 = stock_fwd12.sub(spy_benchmark_fwd12, axis=0)  # broadcast subtract by row

    # Classification label: excess > 0
    y = (excess_fwd12 > 0).astype(int)
    return y

def calculate_enhanced_features(df):
    """
    Calculate enhanced features on long-format dataframe.
    
    Experiments (Oct 2025) showed these features hurt performance:
    - Ranks: -0.007 AUC (Fold 3: -0.031)
    - Interactions: -0.003 AUC
    - Reversal: -0.002 AUC
    Kept for:
    - Future experimentation
    - Different targets (absolute returns)
    - Different models (linear models may benefit)
    Args:
        df: DataFrame (potentially with MultiIndex) containing features
    
    Returns:
        Dictionary of {feature_name: Series}
    """
    enhanced = {}
    
    # Reset index to get access to date/ticker columns
    df_work = df.copy()
    
    # If it has MultiIndex, reset it
    if isinstance(df_work.index, pd.MultiIndex):
        df_work = df_work.reset_index()
    
    # Debug: check what columns we have
    print(f"Columns available: {df_work.columns.tolist()}")
    
    # Find the date column (might be named differently)
    date_col = None
    for col in df_work.columns:
        if 'date' in str(col).lower() or col == 'level_0':
            date_col = col
            break
    
    if date_col is None:
        print("WARNING: Cannot find date column, skipping enhanced features")
        return enhanced
    
    print(f"Using '{date_col}' as date column for grouping")
    # ===== Option 1. use crossectional ranks =====
    rank_cols = ['BookToMarket', 'ROE', 'ROA', 'mom121', 'vol12', 'LogMktCap', 'r12']
    
    for col in rank_cols:
        if col in df_work.columns:
            try:
                enhanced[f'{col}_rank'] = df_work.groupby(date_col)[col].rank(pct=True)
                print(f"  ✓ Added {col}_rank")
            except Exception as e:
                print(f"  ✗ Failed to rank {col}: {e}")
    # ===== Option 2. INTERACTION FEATURES =====
    if 'BookToMarket' in df_work.columns and 'LogMktCap' in df_work.columns:
        enhanced['value_size'] = df_work['BookToMarket'] * df_work['LogMktCap']
        print("  ✓ Added value_size")
    
    if 'mom121' in df_work.columns and 'ROE' in df_work.columns:
        enhanced['mom_quality'] = df_work['mom121'] * df_work['ROE']
        print("  ✓ Added mom_quality")
    
    if 'r12' in df_work.columns and 'vol12' in df_work.columns:
        enhanced['sharpe_12m'] = df_work['r12'] / (df_work['vol12'] + 0.001)
        print("  ✓ Added sharpe_12m")

    # ===== Option 3. monthly reversal =====
    if 'r1' in df_work.columns:
        enhanced['reversal_1m'] = -df_work['r1']
        print("  ✓ Added reversal_1m")

    print(f"\nTotal enhanced features created: {len(enhanced)}")
    
    return enhanced
