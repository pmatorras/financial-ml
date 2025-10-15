import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from .common import DATA_DIR, DEBUG_DIR, FUNDAMENTAL_VARS,FUNDA_KEYS, MARKET_KEYS, CANONICAL_CONCEPTS, get_market_file, get_fundamental_file
from .models import get_models
from pandas.tseries.offsets import BQuarterEnd
import warnings 


def load_market(args):
    csv_filenm =get_market_file(args)
    print("opening", csv_filenm)
    px_all = (pd.read_csv(csv_filenm, index_col=0, parse_dates=True)
                .apply(pd.to_numeric, errors="coerce")
                .sort_index())
    if "SPY" in px_all.columns:
        spy = px_all["SPY"]
        px_m = px_all.drop(columns=["SPY"])
    else:
        px_m = px_all
        # Optional fallback: fetch SPY and align monthly
        import yfinance as yf
        spy = yf.download("SPY", interval="1mo", auto_adjust=True, progress=False)["Close"].reindex(px_m.index).ffill()
    return px_m, spy

def load_fundamentals(args, required_keys=None, keep_unmapped=False):
    csv_filenm = get_fundamental_file(args)
    f = pd.read_csv(csv_filenm, parse_dates=["period_end","filed"])
    f['period_end'] = pd.to_datetime(f['period_end'], errors='coerce')
    f['filed'] = pd.to_datetime(f['filed'], errors='coerce')

    # Derive taxonomy + tag robustly from metric
    m = f['metric'].astype(str)
    has_slash = m.str.contains('/')
    taxonomy = m.where(has_slash, '')
    tag = m.where(~has_slash, m.str.split('/', n=1).str[1])
    tag = tag.where(tag.notna(), m)
    f = f.assign(taxonomy=taxonomy, tag=tag)

    # Build mapping DataFrame from CANONICAL_CONCEPTS
    rows = []
    for canon_key, triples in CANONICAL_CONCEPTS.items():
        for tax, ttag, unit in triples:
            rows.append({'taxonomy': tax, 'tag': ttag, 'unit': unit, 'canonical_key': canon_key})
    map_df = pd.DataFrame(rows)

    # Stage 1: precise match (taxonomy + tag + unit)
    precise = f.merge(map_df, on=['taxonomy','tag','unit'], how='left', suffixes=('','_m1'))

    # Stage 2: fallback by tag + unit (handles plain 'Liabilities' with empty taxonomy)
    fb_map = map_df[['tag','unit','canonical_key']].drop_duplicates()
    fallback = f.merge(fb_map, on=['tag','unit'], how='left', suffixes=('','_m2'))

    # Resolve canonical_key preference: existing -> precise -> fallback
    canon = precise['canonical_key']
    if 'canonical_key' in f.columns:
        canon = f['canonical_key'].where(f['canonical_key'].notna(), canon)
    canon = canon.where(canon.notna(), fallback['canonical_key'])

    out = f.assign(canonical_key=canon)[
        ['ticker','period_end','filed','unit','canonical_key','value','metric','taxonomy','tag']
    ].sort_values(['ticker','canonical_key','period_end','filed'])

    if required_keys is not None:
        out = out[out['canonical_key'].isin(set(required_keys))]
    if not keep_unmapped:
        out = out[out['canonical_key'].notna()]

    if args.debug:
        print(out.columns.tolist())

    if args.debug: out.to_csv(DEBUG_DIR/"fsel.csv")
    return out

def to_monthly_ffill(df, maxdate=None, freq="BM", lag_months=2):
    """
    Minimal monthly duplication by period_end:
    - Groups by (ticker, metric, unit, canonical_key)
    - Resamples at monthly frequency on period_end
    - Forward-fills to duplicate values between quarter ends
    - ADDS LAG to simulate reporting delay
    """
    gcols = ["ticker", "metric", "unit", "canonical_key"]  # <-- ADD canonical_key here

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

    if maxdate is not None:
        maxdate = pd.to_datetime(maxdate)
        out = out[out["period_end"] <= maxdate]

    return out

def require_non_empty(df: pd.DataFrame, name: str, min_rows: int = 1, min_cols: int = 1):
    # .empty is True if any axis has length 0; still False when all-NaN rows exist
    if df is None or df.empty:
        raise ValueError(f"{name} is empty (no rows or no columns).")  # fail fast
    n_rows, n_cols = df.shape
    if n_rows < min_rows or n_cols < min_cols:
        raise ValueError(f"{name} too small: shape={df.shape}, expected at least ({min_rows}, {min_cols}).")

def widenVariable2(monthly, prices, canonical_key):
    monthly["period_end"] = pd.to_datetime(monthly["period_end"], errors="coerce")

    wide = (
        monthly.loc[monthly["canonical_key"] == canonical_key,
                   ["period_end", "ticker", "value"]]
        .pivot(index="period_end", columns="ticker", values="value")
        .sort_index()
    )
    # Align to prices shape: same index and same columns order
    require_non_empty(wide, f"wide_{canonical_key}")
    wide = wide.reindex(index=prices.index, columns=prices.columns)
    require_non_empty(wide, "wide")
    return wide
def widenVariable(monthly, prices, m="us-gaap/Assets", u="USD"):
    monthly["period_end"] = pd.to_datetime(monthly["period_end"], errors="coerce")

    wide = (
        monthly.loc[(monthly["metric"] == m) & (monthly["unit"] == u),
                    ["period_end","ticker","value"]]
            .pivot(index="period_end", columns="ticker", values="value")
            .sort_index()
    )
    # Align to prices shape: same index and same columns order
    require_non_empty(monthly.loc[(monthly["metric"] == m) & (monthly["unit"] == u),
                    ["period_end","ticker","value"]], m)
    wide = wide.reindex(index=prices.index, columns=prices.columns)
    require_non_empty(wide, "wide")
    return wide
def ensure_shares(monthly, prices):
    # Accept both aliases explicitly via metric
    share_metrics = {
        "us-gaap/CommonStockSharesOutstanding",
        "dei/EntityCommonStockSharesOutstanding",
    }

    shares = (
        monthly.loc[
            monthly["metric"].isin(share_metrics),
            ["period_end", "ticker", "metric", "unit", "value"],
        ]
        .sort_values(["ticker", "period_end", "metric"])
        .drop_duplicates(subset=["ticker", "period_end"], keep="last")  # prefer later metric if both present
        .pivot(index="period_end", columns="ticker", values="value")
        .reindex(index=prices.index, columns=prices.columns)
    )

    # Ensure numeric
    return pd.to_numeric(shares.stack(), errors="coerce").unstack()
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

def safe_div(numer, denom):
    #return nan if negative/zero denominators
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom
    return out.where((denom > 0) & np.isfinite(out))



def train(args):
    prices, spy_benchmark = load_market(args)
    require_non_empty(prices, "prices")

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    ret_1m = prices.pct_change(1, fill_method=None) #1 month change
    ret_12m = prices.pct_change(12, fill_method=None) #change in 12 months
    mom_12_1 = (1 + ret_12m) / (1 + ret_1m) - 1 #month on month momentum
    vol_3m = ret_1m.rolling(3).std() #volumes
    vol_12m = ret_1m.rolling(12).std()

    ret_1m.to_csv(DEBUG_DIR/"testret.csv")

    input_vars = [prices, ret_1m, ret_12m, mom_12_1, vol_3m, vol_12m]
    market_keys = MARKET_KEYS
    funda_keys = []
    df_keys = market_keys.copy()          # safe copy for concat keys

    if args.use_fundamentals:
        funda_keys = FUNDA_KEYS
        fundamentals = load_fundamentals(args)
        if args.debug: fundamentals.to_csv(DEBUG_DIR/"funda.csv")
        monthly = to_monthly_ffill(fundamentals, maxdate="2025-12-31", freq="BME")
        if args.debug: monthly.to_csv(DEBUG_DIR/"monthly.csv")
        require_non_empty(monthly, "monthly")
        shares = ensure_shares(monthly, prices)
        if args.debug: shares.to_csv(DEBUG_DIR/"shares.csv")
        require_non_empty(shares, "SharesOutstanding")
        shares.name = "SharesOutstanding"

        # 2) Non-share concepts by exact metric (unchanged behavior)
        input_vars = [prices, ret_1m, ret_12m, mom_12_1, vol_3m, vol_12m]
        df_keys = MARKET_KEYS.copy()

        for tax, tag, unit in FUNDAMENTAL_VARS:
            m = f"{tax}/{tag}"
            print(tax)
            wide = widenVariable(monthly, prices, m, unit)
            input_vars.append(wide)
            df_keys.append(tag if tag not in ("CommonStockSharesOutstanding","EntityCommonStockSharesOutstanding") else "IGNORED")

        # Inject the canonical shares column explicitly once
        input_vars.append(shares)
        df_keys.append("SharesOutstanding")

    input_keys=market_keys+funda_keys
    if len(df_keys) != len(input_vars):
        print("keys and variables not the same size!")
        exit()

    if args.debug:
        for k, df in zip(df_keys, input_vars):
            print(k, type(df.index), df.index.min(), df.index.max(), df.shape)
    feat = pd.concat(input_vars, axis=1, keys=df_keys)
    if args.debug: feat.to_csv(DEBUG_DIR/"feat.csv")
    
    # Label: 12m forward total return > 0
    stock_fwd12 = (prices.shift(-12) / prices) - 1  # (future / now) - 1
    spy_benchmark_fwd12 = (spy_benchmark.shift(-12) / spy_benchmark) - 1   # spy_benchmark forward 12m return (Series)
    excess_fwd12 = stock_fwd12.sub(spy_benchmark_fwd12, axis=0)  # broadcast subtract by row

    # Classification label: excess > 0
    y = (excess_fwd12 > 0).astype(int)


    # Align and stack panel to long format
    feat_long = feat.stack(level=1,future_stack=True)
    if "canonical_key" not in feat_long.columns:
        feat_long["canonical_key"] = ""
    require_non_empty(feat_long, "feat_long")
    if args.debug: print(feat_long.keys(), feat_long)
    if args.use_fundamentals:
        price = feat_long["ClosePrice"].astype(float)
        if args.debug: feat_long.to_csv(DEBUG_DIR/"feat_long.csv")
        #n_shares = feat_long['us-gaap/CommonStockSharesOutstanding'].astype(float)
        n_shares = feat_long["SharesOutstanding"].astype(float)
        if args.debug: n_shares.to_csv(DEBUG_DIR/"nshares.csv")
        liabilities = feat_long["Liabilities"].astype(float)
        assets = feat_long["Assets"].astype(float)
        # Size: log market cap at current date
        market_cap = price * n_shares
        mcap = market_cap.astype(float)
        bad = ~np.isfinite(mcap) | (mcap <= 0)
        if args.debug: print("nonpositive_or_missing:", bad.sum())           # count of bad inputs
        if args.debug: print(mcap[bad].head())                                # spot-check values
        if args.debug: mcap.to_csv(DEBUG_DIR/"marketcap.csv")
        feat_long['LogMktCap'] = compute_log_mktcap(
            price,
            n_shares,
            name='LogMktCap'
)

        dates = pd.Series(feat_long.index.get_level_values('Date'))
        is_bqe = (dates == pd.DatetimeIndex([BQuarterEnd().rollback(d) for d in dates])).values
        bq_roll = pd.DatetimeIndex([BQuarterEnd().rollback(d) for d in dates])

        #Take quarter ends, and do the 4 quarter lags
        assets_qe = assets[is_bqe]
        shares_qe = n_shares[is_bqe]
        assets_lag4 = assets_qe.groupby(level=1).shift(4)
        shares_lag4 = shares_qe.groupby(level=1).shift(4)
        inv_qe = (assets_qe - assets_lag4) / assets_lag4
        iss_qe = (shares_qe - shares_lag4) / shares_lag4
        inv_qe = inv_qe.replace([np.inf, -np.inf], np.nan)
        iss_qe = iss_qe.replace([np.inf, -np.inf], np.nan)
        if args.debug:
            check = pd.DataFrame({
                'Date': dates,
                'BQ_Rollback': bq_roll,
                'Is_BQ_End': is_bqe,
            }).set_index('Date')
            if args.debug: check.to_csv(DEBUG_DIR/"check_dates.csv", mode='w')
        # Write quarterly results back and forward-fill to months for the asset growth and netshare insurance"    
        feat_long['AssetGrowth'] = np.nan
        feat_long.loc[inv_qe.index, 'AssetGrowth'] = inv_qe
        feat_long['AssetGrowth'] = feat_long['AssetGrowth'].groupby(level=1).ffill()

        feat_long['NetShareIssuance'] = np.nan
        feat_long.loc[iss_qe.index, 'NetShareIssuance'] = iss_qe
        feat_long['NetShareIssuance'] = feat_long['NetShareIssuance'].groupby(level=1).ffill()

        eq = feat_long["StockholdersEquity"].astype(float)
        netIncome = feat_long["NetIncomeLoss"]
        # Compute lag and average equity first
        feat_long["Equity_prev"] = eq.groupby(level=1).shift(1)
        feat_long["AvgEquity"] = (eq + feat_long["Equity_prev"]) / 2.0

        #market equity
        feat_long["MarketEquity"] = price * n_shares
        feat_long["BookToMarket"] = safe_div(eq,feat_long["MarketEquity"])
        feat_long["ROE"]= safe_div(netIncome,feat_long["AvgEquity"])
        feat_long["ROA"]= safe_div(netIncome,assets)

        feat_long["NetMargin"]=safe_div(netIncome,feat_long["Revenues"])
        feat_long["Leverage"]=safe_div(liabilities,assets)
        if args.debug: print(feat_long.keys(), feat_long["ROE"])

    #Add the Y to the datafrane
    y_long = y.stack().rename("y")
    df = feat_long.join(y_long, how="inner")
    print(f"before first dropna: {len(df)} rows, {df.index.get_level_values(1).nunique()} unique tickers")
    #df = df.dropna()
    print(f"After first dropna: {len(df)} rows, {df.index.get_level_values(1).nunique()} unique tickers")
    require_non_empty(df, "feat_join_ylong")
    # Train/validation split: expanding window via TimeSeriesSplit
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]:"date",df.columns[1]:"ticker"})
    # Guards: types, sort, chronology
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    assert df["date"].is_monotonic_increasing, "Dates must be sorted ascending"  # guard
    require_non_empty(df, "df_sorted")
    # Optional: ensure no NaNs remain in features/label before splitting


    # Show which features in input_keys have the most NaN
    print("\nNaN counts in input_keys:")
    nan_in_input_keys = df[input_keys].isnull().sum().sort_values(ascending=False)
    print(nan_in_input_keys[nan_in_input_keys > 0])

    # Show which tickers are getting dropped
    if 'ticker' in df.columns:
        tickers_before = set(df['ticker'].unique())
        df_after = df.dropna(subset=input_keys)
        tickers_after = set(df_after['ticker'].unique())
        dropped_tickers = tickers_before - tickers_after
        print(f"\nTickers dropped: {len(dropped_tickers)} out of {len(tickers_before)}")
        print(f"Sample dropped tickers: {sorted(list(dropped_tickers))[:20]}")
        print(f"Sample surviving tickers: {sorted(list(tickers_after))[:20]}")
        
        # For a few dropped tickers, show which features are NaN
        if dropped_tickers:
            sample_dropped = list(dropped_tickers)[:3]
            print(f"\nWhy these tickers were dropped:")
            for ticker in sample_dropped:
                ticker_rows = df[df['ticker'] == ticker]
                missing_features = ticker_rows[input_keys].isnull().sum()
                missing_features = missing_features[missing_features > 0]
                print(f"  {ticker}: missing {len(missing_features)} features")
                print(f"    {missing_features.head(10).to_dict()}")


    df = df.dropna(subset=input_keys)
    print(f"\nafter second dropna: {len(df)} rows")
    if 'ticker' in df.columns:
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        print(f"  Sample tickers: {sorted(df['ticker'].unique())[:20]}")
        
    print(f"DTE Leverage: {len(df[df['ticker']=='DTE'])} total, {df[df['ticker']=='DTE']['Leverage'].notna().sum()} non-null")
    exit()
    print("Input variables to train", input_keys)
    X = df[input_keys].to_numpy()
    Y = df["y"].to_numpy()

    dates = pd.to_datetime(df["date"])
    order = np.argsort(dates.values)
    X, Y = X[order], Y[order]
    unique_dates = np.array(sorted(df["date"].unique()))

    tscv = TimeSeriesSplit(n_splits=3,test_size=36,gap=1)

    pred_rows = []
    models = get_models()
    trained_models = {}
    print(f"Available models [{len(models)}]: {', '.join(sorted(models))}")
    for name, pipe in models.items():
        aucs_test = []
        aucs_train = []
        for  split_id, (tr_d, te_d) in enumerate(tscv.split(unique_dates), 1):
            train_dates = set(unique_dates[tr_d])
            test_dates  = set(unique_dates[te_d])

            tr_mask = df["date"].isin(train_dates).to_numpy()
            te_mask = df["date"].isin(test_dates).to_numpy()

            Xtrain, Xtest = X[tr_mask], X[te_mask]
            Ytrain, Ytest = Y[tr_mask], Y[te_mask]
            # Log fold ranges and test class balance
            tr_start, tr_end = min(train_dates), max(train_dates)
            te_start, te_end = min(test_dates), max(test_dates)
            classes, counts = np.unique(Ytest, return_counts=True)
            if args.debug:
                print(f"[{name} | Fold {split_id}] "
                    f"Train {tr_start.date()} → {tr_end.date()} | "
                    f"Test {te_start.date()} → {te_end.date()} | "
                    #f"Test class counts {dict(zip(classes, counts))}"
                    )
            print(f"[{name} | Fold {split_id}] "
                  f"Train {tr_start.date()} → {tr_end.date()} | "
                  f"Test {te_start.date()} → {te_end.date()} | "
                    )
            pipe.fit(Xtrain, Ytrain)
            p_train = pipe.predict_proba(Xtrain)[:,1]
            p_test = pipe.predict_proba(Xtest)[:,1]
            aucs_train.append(roc_auc_score(Ytrain, p_train))
            aucs_test.append(roc_auc_score(Ytest, p_test))
            fold_df = df.loc[te_mask, ["date","ticker"]].copy()
            fold_df["y_true"] = Ytest
            fold_df["y_prob"] = p_test
            fold_df["y_pred"] = (p_test >= 0.5).astype(int)
            fold_df["fold"] = split_id
            fold_df["model"] = name
            pred_rows.append(fold_df)

        trained_models[name] = pipe
        print("Baseline logistic train AUC ("+name+"):", np.round(aucs_train, 3).tolist())
        print("Baseline logistic test  AUC ("+name+"):", np.round(aucs_test, 3).tolist())

    pred_df = pd.concat(pred_rows, ignore_index=True)
    predpath = DATA_DIR/"oof_predictions.csv"
    pred_df.to_csv(predpath, index=False)

    from financial_ml.feature_importance import analyze_feature_importance
    analyze_feature_importance(
        models_dict=trained_models, # Dict with model names as keys, trained pipelines as values
        X=X,                       # numpy array shape (n_samples, n_features)
        y=Y,                       # numpy array shape (n_samples,)
        feature_names=input_keys   # List like ['ClosePrice', 'r1', 'r12', 'mom121', ...]
    )

    