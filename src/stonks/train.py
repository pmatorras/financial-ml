import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import yfinance as yf
import common



px_all = (pd.read_csv(common.SP500_MARKET_FILE, index_col=0, parse_dates=True)
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

px_m = px_m.apply(pd.to_numeric, errors="coerce")
px_m = px_m.sort_index()
ret_1m = px_m.pct_change(1) #1 month change
ret_12m = px_m.pct_change(12) #change in 12 months
mom_12_1 = px_m.pct_change(12) - px_m.pct_change(1)  # simple momentum variant
vol_3m = ret_1m.rolling(3).std() #volumes
vol_12m = ret_1m.rolling(12).std()

input_vars = [ret_1m, ret_12m, mom_12_1, vol_3m, vol_12m]
input_keys = ["r1", "r12", "mom121","vol3","vol12"]
if len(input_keys) != len(input_vars):
    print("keys and variables not the same size!")
    exit()
feat = pd.concat(input_vars, axis=1, keys=input_keys)

# Label: 12m forward total return > 0
stock_fwd12 = px_m.pct_change(12).shift(-12)     # stock forward 12m return
spy_fwd12   = spy.pct_change(12).shift(-12)      # SPY forward 12m return (Series)
excess_fwd12 = stock_fwd12.sub(spy_fwd12, axis=0)  # broadcast subtract by row

# Classification label: excess > 0
y = (excess_fwd12 > 0).astype(int)


# Align and stack panel to long format
feat_long = feat.stack(level=1,future_stack=True)
y_long = y.stack().rename("y")
df = feat_long.join(y_long, how="inner").dropna()

# Train/validation split: expanding window via TimeSeriesSplit
df = df.reset_index()
df = df.rename(columns={df.columns[0]:"date",df.columns[1]:"ticker"})
X = df[input_keys].values
Y = df["y"].values
dates = pd.to_datetime(df["date"])
order = np.argsort(dates.values)
X, Y = X[order], Y[order]
tscv = TimeSeriesSplit(n_splits=5)

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
}
for name, pipe in models.items():
    aucs_test = []
    aucs_train = []
    for train, test in tscv.split(X):
        Xtrain, Xtest, Ytrain, Ytest = X[train], X[test], Y[train], Y[test]
        # Baseline logistic
        pipe.fit(Xtrain, Ytrain)
        p_train = pipe.predict_proba(Xtrain)[:,1]
        p_test = pipe.predict_proba(Xtest)[:,1]
        aucs_train.append(roc_auc_score(Ytrain, p_train))
        aucs_test.append(roc_auc_score(Ytest, p_test))

    print("Baseline logistic train AUC ("+name+"):", np.round(aucs_train, 3).tolist())
    print("Baseline logistic test  AUC ("+name+"CV):", np.round(aucs_test, 3).tolist())

# 7) Random forest
#rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=42)
#rf.fit(X[train], Y[train])
#p_rf = rf.predict_proba(X[test])[:,1]
#print("RF last fold AUC:", roc_auc_score(Y[test], p_rf))
