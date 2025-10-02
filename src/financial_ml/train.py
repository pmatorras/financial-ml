import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .common import SP500_MARKET_FILE, SP500_MARKET_TEST, DATA_DIR, FUNDAMENTAL_VARS, SP500_FUNDA_FILE, SP500_FUNDA_TEST, TEST_DIR,FUNDA_KEYS, MARKET_KEYS
from pandas.tseries.offsets import BQuarterEnd
import warnings 
sanitize = FunctionTransformer(
    lambda X: np.where(np.isfinite(X), X, np.nan), validate=False
)
models = {
    "logreg_l2": Pipeline([
        ("sanitize", sanitize),                    # replace ±inf with NaN
        ("impute", SimpleImputer(strategy="median")),  # handle NaN
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    ]),
    "logreg_l1": Pipeline([
        ("sanitize", sanitize),                    # replace ±inf with NaN
        ("impute", SimpleImputer(strategy="median")),  # handle NaN
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", penalty="l1",
                                C=0.5, max_iter=5000, class_weight="balanced"))
    ])
    ,
    "rf": Pipeline([
        ("sanitize", sanitize),                    # replace ±inf with NaN
        ("impute", SimpleImputer(strategy="median")),  # handle NaN
        ("scaler", "passthrough"),  # trees don"t need scaling
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=None,
                                    min_samples_leaf=5, n_jobs=-1, class_weight="balanced_subsample"))
    ])
}

def load_market(args):
    csv_filenm = SP500_MARKET_TEST if args.test else SP500_MARKET_FILE
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
def load_fundamentals(args):
    csv_filenm = SP500_FUNDA_TEST if args.test else SP500_FUNDA_FILE
    print("opening", csv_filenm)
    f = pd.read_csv(csv_filenm, parse_dates=["period_end","filed"])
    f["metric"] = f["metric"].astype("string")
    if args.debug: print(f.keys())
    f[["taxonomy","tag"]] = f["metric"].str.split("/", n=1, expand=True)

    targets = pd.DataFrame(FUNDAMENTAL_VARS, columns=["taxonomy","tag","unit"])
    f_sel = f.merge(targets, on=["taxonomy","tag","unit"], how="inner")
    if args.debug: print(f_sel.keys())
    f_sel = f_sel.drop(columns=["taxonomy","tag"])
    return f_sel

def to_monthly_ffill(df, maxdate=None, freq="BM"):
    """
    Minimal monthly duplication by period_end:
    - Groups by (ticker, metric, unit)
    - Resamples at monthly frequency on period_end
    - Forward-fills to duplicate values between quarter ends
    """
    gcols = ["ticker", "metric", "unit"]
    out = (
        df.copy()
          .assign(period_end=pd.to_datetime(df["period_end"], errors="coerce", utc=True).dt.tz_convert(None))
          .sort_values(gcols + ["period_end"])
          .set_index("period_end")
          .groupby(gcols)["value"]
          .resample(freq)
          .ffill()
          .rename("value")
          .reset_index()
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

    ret_1m.to_csv(TEST_DIR/"testret.csv")

    input_vars = [prices, ret_1m, ret_12m, mom_12_1, vol_3m, vol_12m]
    market_keys = MARKET_KEYS
    funda_keys = []
    df_keys = market_keys.copy()          # safe copy for concat keys
    if args.trainfundamentals:
        funda_keys = FUNDA_KEYS
        fundamentals = load_fundamentals(args)
        monthly = to_monthly_ffill(fundamentals, maxdate="2025-12-31", freq="BME")
        require_non_empty(monthly, "monthly")
        for fundavar in FUNDAMENTAL_VARS:
            fundamental = "us-gaap/"+fundavar[1]
            unit = fundavar[2]
            if args.debug:
                print(f"processing: {fundamental} [{unit}]")
            widen = widenVariable(monthly, prices, fundamental, unit)    
            widen.to_csv(TEST_DIR/"widen.csv")
            df_keys.append(fundamental)
            input_vars.append(widen)
    input_keys=market_keys+funda_keys   
    if len(df_keys) != len(input_vars):
        print("keys and variables not the same size!")
        exit()

    if args.debug:
        for k, df in zip(df_keys, input_vars):
            print(k, type(df.index), df.index.min(), df.index.max(), df.shape)
    feat = pd.concat(input_vars, axis=1, keys=df_keys)
    feat.to_csv(TEST_DIR/"feat.csv")
    
    # Label: 12m forward total return > 0
    stock_fwd12 = prices.pct_change(12, fill_method=None).shift(-12)     # stock forward 12m return
    spy_benchmark_fwd12   = spy_benchmark.pct_change(12).shift(-12)      # spy_benchmark forward 12m return (Series)
    excess_fwd12 = stock_fwd12.sub(spy_benchmark_fwd12, axis=0)  # broadcast subtract by row

    # Classification label: excess > 0
    y = (excess_fwd12 > 0).astype(int)


    # Align and stack panel to long format
    feat_long = feat.stack(level=1,future_stack=True)
    require_non_empty(feat_long, "feat_long")
    if args.debug: print(feat_long.keys(), feat_long)
    if args.trainfundamentals:
        price = feat_long["ClosePrice"].astype(float)
        shares = feat_long['us-gaap/CommonStockSharesOutstanding'].astype(float)
        assets = feat_long["us-gaap/Liabilities"].astype(float)
        # Size: log market cap at current date
        market_cap = price * shares
        mcap = market_cap.astype(float)
        bad = ~np.isfinite(mcap) | (mcap <= 0)
        print("nonpositive_or_missing:", bad.sum())           # count of bad inputs
        print(mcap[bad].head())                                # spot-check values
        if args.debug: mcap.to_csv(TEST_DIR/"marketcap.csv")
        feat_long['LogMktCap'] = compute_log_mktcap(
            feat_long['ClosePrice'],
            feat_long['us-gaap/CommonStockSharesOutstanding'],
            name='LogMktCap'
)

        dates = pd.Series(feat_long.index.get_level_values('Date'))
        is_bqe = (dates == pd.DatetimeIndex([BQuarterEnd().rollback(d) for d in dates])).values
        bq_roll = pd.DatetimeIndex([BQuarterEnd().rollback(d) for d in dates])

        #Take quarter ends, and do the 4 quarter lags
        assets_qe = assets[is_bqe]
        shares_qe = shares[is_bqe]
        assets_lag4 = assets_qe.groupby(level=1).shift(4)
        shares_lag4 = shares_qe.groupby(level=1).shift(4)
        inv_qe = (assets_qe - assets_lag4) / assets_lag4
        iss_qe = (shares_qe - shares_lag4) / shares_lag4
        if args.debug:
            check = pd.DataFrame({
                'Date': dates,
                'BQ_Rollback': bq_roll,
                'Is_BQ_End': is_bqe,
            }).set_index('Date')
            check.to_csv(TEST_DIR/"check_dates.csv", mode='x')
        # Write quarterly results back and forward-fill to months for the asset growth and netshare insurance"    
        feat_long['AssetGrowth'] = np.nan
        feat_long.loc[inv_qe.index, 'AssetGrowth'] = inv_qe
        feat_long['AssetGrowth'] = feat_long['AssetGrowth'].groupby(level=1).ffill()

        feat_long['NetShareIssuance'] = np.nan
        feat_long.loc[iss_qe.index, 'NetShareIssuance'] = iss_qe
        feat_long['NetShareIssuance'] = feat_long['NetShareIssuance'].groupby(level=1).ffill()



        feat_long["MarketEquity"] = price - shares
        feat_long["BookToMarket"] = feat_long["us-gaap/StockholdersEquity"] /feat_long["MarketEquity"]
        eq = feat_long["us-gaap/StockholdersEquity"].astype(float)
        feat_long["Equity_prev"] = eq.groupby(level=1).shift(1)
        feat_long["AvgEquity"] = (feat_long["us-gaap/StockholdersEquity"] + feat_long["Equity_prev"]) / 2.0
        feat_long["ROE"]=feat_long["us-gaap/NetIncomeLoss"]-feat_long["AvgEquity"]
        feat_long["ROA"]=feat_long["us-gaap/NetIncomeLoss"]-feat_long["us-gaap/Assets"]
        feat_long["NetMargin"]=feat_long["us-gaap/NetIncomeLoss"]-feat_long["us-gaap/Revenues"]
        feat_long["Leverage"]=assets-feat_long["us-gaap/Assets"]
        if args.debug: print(feat_long.keys(), feat_long["ROE"])

    #Add the Y to the datafrane
    y_long = y.stack().rename("y")

    df = feat_long.join(y_long, how="inner").dropna()
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

    df = df.dropna(subset=input_keys)
    print("input keys", input_keys)
    X = df[input_keys].to_numpy()
    Y = df["y"].to_numpy()

    dates = pd.to_datetime(df["date"])
    order = np.argsort(dates.values)
    X, Y = X[order], Y[order]
    unique_dates = np.array(sorted(df["date"].unique()))

    tscv = TimeSeriesSplit(n_splits=5,test_size=36)

    pred_rows = []
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
        print("Baseline logistic train AUC ("+name+"):", np.round(aucs_train, 3).tolist())
        print("Baseline logistic test  AUC ("+name+"):", np.round(aucs_test, 3).tolist())
    pred_df = pd.concat(pred_rows, ignore_index=True)
    predpath = DATA_DIR/"oof_predictions.csv"
    pred_df.to_csv(predpath, index=False)

