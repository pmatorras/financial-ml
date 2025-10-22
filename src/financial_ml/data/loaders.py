import pandas as pd
from financial_ml.utils.paths import get_market_file, get_fundamental_file
from financial_ml.utils.config import DEBUG_DIR

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

    # If canonical_key already exists in CSV, use it directly
    if 'canonical_key' in f.columns:
        out = f[['ticker','period_end','filed','unit','canonical_key','value','metric']].copy()
        out = out.sort_values(['ticker','canonical_key','period_end','filed'])
        
        if required_keys is not None:
            out = out[out['canonical_key'].isin(set(required_keys))]
        if not keep_unmapped:
            out = out[out['canonical_key'].notna()]
        
        if args.debug: 
            print(out.columns.tolist())
            out.to_csv(DEBUG_DIR/"fsel.csv")
        
        return out