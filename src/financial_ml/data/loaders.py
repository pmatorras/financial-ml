'''
Load all information from markets, fundamentals or sentiment
'''

import pandas as pd
from financial_ml.utils.paths import get_market_file, get_fundamental_file, get_sentiment_file
from financial_ml.utils.config import DEBUG_DIR

def normalize_ticker(ticker):
    """
    Normalize ticker symbols to handle different formats.
    BRK-B (yfinance) -> BRK.B (SEC standard)
    BRK.B (SEC) -> BRK.B (keep as is)
    """
    if pd.isna(ticker):
        return ticker
    return str(ticker).replace('-', '.')


def load_market(args):
    csv_filenm =get_market_file(args)
    print("opening", csv_filenm)
    px_all = (pd.read_csv(csv_filenm, index_col=0, parse_dates=True)
                .apply(pd.to_numeric, errors="coerce")
                .sort_index())
    # NORMALIZE TICKER COLUMN NAMES
    px_all.columns = [normalize_ticker(col) for col in px_all.columns]
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
    dtype_spec = {
        'cik': str,  # Force CIK to string
        'source_cik': str,  # Force source_cik to string
        'ticker': str,
        'unit': str,
        'metric': str,
    }
    f = pd.read_csv(csv_filenm, parse_dates=["period_end","filed"],  dtype=dtype_spec )
    f['period_end'] = pd.to_datetime(f['period_end'], errors='coerce')
    f['filed'] = pd.to_datetime(f['filed'], errors='coerce')
    if 'ticker' in f.columns:
        f['ticker'] = f['ticker'].apply(normalize_ticker)
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
    
def load_sentiment(args):
    """
    Load sentiment indicators (VIX, etc.) from CSV
    
    Args:
        args: Namespace (for test mode, etc.)
    
    Returns:
        pd.DataFrame with sentiment indicators indexed by date
    """
    csv_filenm = get_sentiment_file(args)
    print(f"Loading sentiment data from {csv_filenm}")
    
    sentiment = pd.read_csv(csv_filenm, index_col=0, parse_dates=True)
    sentiment = sentiment.apply(pd.to_numeric, errors='coerce').sort_index()
    
    return sentiment
