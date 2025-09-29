# pip install requests pandas yfinance
import time
from io import StringIO
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import pandas as pd
from common import SP500_NAMES_FILE, SP500_FUNDA_FILE,SP500_FUNDA_TEST

# SEC guidance: include a descriptive User-Agent with contact email and keep request rate modest
UA = "ResearchBot/1.0 (contact@example.com)"  
BASE = "https://data.sec.gov"

def session_with_retries():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

sess = session_with_retries()

def get_json(url, params=None, sleep=0.12):
    r = sess.get(url, params=params, timeout=30)
    r.raise_for_status()
    if sleep: time.sleep(sleep)
    return r.json()

# Helper to extract one tag/unit into a tidy PIT series
def extract_tag_pit(cf_json, taxonomy, tag, unit):
    try:
        facts = cf_json["facts"][taxonomy][tag]["units"][unit]
    except KeyError:
        return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])
    df = pd.DataFrame(facts)
    # normalize and parse dates
    df = df.rename(columns={"end":"period_end","val":"value"})
    for col in ["period_end","filed","start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["metric"] = f"{taxonomy}/{tag}"
    df["unit"] = unit
    # keep only dated facts
    return df[["period_end","filed","value","metric","unit"]].dropna(subset=["filed"])

# Select a few common metrics and units
targets = [
    ("us-gaap","Assets","USD"),
    ("us-gaap","Liabilities","USD"),
    ("us-gaap","StockholdersEquity","USD"),
    ("us-gaap","Revenues","USD"),
    ("us-gaap","NetIncomeLoss","USD"),
    ("us-gaap","EarningsPerShareDiluted","USD/share"),
    ("us-gaap","CommonStockSharesOutstanding","shares"), #Careful with double counting this (see https://www.perplexity.ai/search/i-am-developing-a-framework-th-6QC.Sc1JS1GjJQBfELScNw#5)
]
def dropDuplicateInfo(_df, key_cols):
    '''Drop data when duplicated'''
    df_last = _df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
    return df_last

def fetch_facts_latest_for_cik(cik, ticker, targets):
    cf = get_json(f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json")
    series = [extract_tag_pit(cf, *t) for t in targets]
    series_nonempty = [df for df in series if not df.empty]
    cols = ["period_end","filed","value","metric","unit"]
    facts_long = (
        pd.concat(series_nonempty, ignore_index=True)
        if series_nonempty else
        pd.DataFrame(columns=cols)
    )
    facts_latest = dropDuplicateInfo(facts_long, ["metric","unit","period_end"])
    facts_latest["cik"] = cik
    facts_latest["ticker"] = ticker
    return facts_latest

# Example CIK↔ticker list; in production, build from SEC company_tickers.json+
def fetch_sp500_list(filepath, force_download=False, url=None, headers=None):
    '''Download SP500 list of companies'''
    if force_download or os.path.getsize(filepath)<1:
        print("Downloading the sp500 list")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Ensure we got a 200 OK

        # Parse HTML content with pandas
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        df.to_csv(filepath)
        print(f"Saved to {filepath}")

    else:   
        print(f"File {filepath} should be already available")
        df = pd.read_csv(filepath)
    return df
df = fetch_sp500_list(SP500_NAMES_FILE, False)
print(df)
pairs = list(
    df.loc[:, ["CIK", "Symbol"]]
      .assign(CIK=lambda x: x["CIK"].astype(str).str.zfill(10))
      .to_records(index=False)
)
# If NumPy recarray tuples are not desired, cast explicitly:
universe = [ (cik, sym) for cik, sym in pairs ]
fundamentals_file= SP500_FUNDA_FILE
test=True
if test:
    fundamentals_file = SP500_FUNDA_TEST
    universe = [
    ("0000066740","MMM"),
    ("0000320193","AAPL"),
    ("0000789019","MSFT"),
    ]
print(fundamentals_file)
all_facts = []
for cik, ticker in universe:
    print(ticker)
    try:
        df_i = fetch_facts_latest_for_cik(cik, ticker, targets)
        all_facts.append(df_i)
    except Exception as e:
        print(f"skip {cik} {ticker}: {e}")

facts_latest_all = pd.concat(all_facts, ignore_index=True)

# Optional: deterministic ordering
facts_latest_all = facts_latest_all.sort_values(
    ["ticker","metric","unit","period_end","filed"]
).reset_index(drop=True)

# Write once to CSV
print("Saving file to:", fundamentals_file)
print(facts_latest_all)
exit()
facts_latest_all.to_csv(fundamentals_file, index=False)
print("Finished.")


