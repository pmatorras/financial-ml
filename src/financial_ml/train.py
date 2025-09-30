import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import yfinance as yf
from .common import SP500_MARKET_FILE, SP500_MARKET_TEST, DATA_DIR, FUNDAMENTAL_VARS, SP500_FUNDA_FILE, SP500_FUNDA_TEST
import argparse

models = {
    "logreg_l2": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    ]),
    "logreg_l1": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", penalty="l1",
                                C=0.5, max_iter=5000, class_weight="balanced"))
    ])
    ,
    "rf": Pipeline([
        ("scaler", "passthrough"),  # trees don't need scaling
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
    print(f.keys())
    f[["taxonomy","tag"]] = f["metric"].str.split("/", n=1, expand=True)

    targets = pd.DataFrame(FUNDAMENTAL_VARS, columns=["taxonomy","tag","unit"])
    f_sel = f.merge(targets, on=["taxonomy","tag","unit"], how="inner")
    print(f_sel.keys())
    f_sel = f_sel.drop(columns=["taxonomy","tag"])
    return f_sel
def train(args):
    fundamentals = load_fundamentals(args)
    exit()
    prices, spy_benchmark = load_market(args)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    ret_1m = prices.pct_change(1) #1 month change
    ret_12m = prices.pct_change(12) #change in 12 months
    mom_12_1 = prices.pct_change(12) - prices.pct_change(1)  # simple momentum variant
    vol_3m = ret_1m.rolling(3).std() #volumes
    vol_12m = ret_1m.rolling(12).std()

    input_vars = [ret_1m, ret_12m, mom_12_1, vol_3m, vol_12m]
    input_keys = ["r1", "r12", "mom121","vol3","vol12"]
    if len(input_keys) != len(input_vars):
        print("keys and variables not the same size!")
        exit()
    feat = pd.concat(input_vars, axis=1, keys=input_keys)

    # Label: 12m forward total return > 0
    stock_fwd12 = prices.pct_change(12).shift(-12)     # stock forward 12m return
    spy_benchmark_fwd12   = spy_benchmark.pct_change(12).shift(-12)      # spy_benchmark forward 12m return (Series)
    excess_fwd12 = stock_fwd12.sub(spy_benchmark_fwd12, axis=0)  # broadcast subtract by row

    # Classification label: excess > 0
    y = (excess_fwd12 > 0).astype(int)


    # Align and stack panel to long format
    feat_long = feat.stack(level=1,future_stack=True)
    y_long = y.stack().rename("y")
    df = feat_long.join(y_long, how="inner").dropna()

    # Train/validation split: expanding window via TimeSeriesSplit
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]:"date",df.columns[1]:"ticker"})
    # Guards: types, sort, chronology
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    assert df["date"].is_monotonic_increasing, "Dates must be sorted ascending"  # guard

    # Optional: ensure no NaNs remain in features/label before splitting
    df = df.dropna(subset=input_keys)
    X = df[["r1","r12","mom121","vol3","vol12"]].to_numpy()
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
            if args.test:
                print(f"[{name} | Fold {split_id}] "
                    f"Train {tr_start.date()} → {tr_end.date()} | "
                    f"Test {te_start.date()} → {te_end.date()} | "
                    f"Test class counts {dict(zip(classes, counts))}")

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


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GDP and Inflation for selected countries")
    parser.add_argument("-nt", "--newtable", action="store_true", help="Update sp500 table")    
    parser.add_argument("-ni", "--newinfo", action="store_true", help="Update sp500 financial information") 
    parser.add_argument("-t" , "--test"   , action="store_true", help="Test on a smaller subset of 50")    
   
    args = parser.parse_args()
    main()
'''