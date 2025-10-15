import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from .common import SP500_NAMES_FILE, SP500_FUNDA_FILE,SP500_FUNDA_TEST, FUNDAMENTAL_VARS, CANONICAL_CONCEPTS, SP500_FUNDA_DEBUG, get_fundamental_file
from .markets import test_subset

# SEC guidance: include a descriptive User-Agent with contact email and keep request rate modest
UA = "ResearchBot/1.0 (contact@example.com)"  
BASE = "https://data.sec.gov"

def get_json(url, params=None, sleep=0.12):
    '''Create a requests session, and fetch json data from the url'''
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    r = sess.get(url, params=params, timeout=30)
    r.raise_for_status()
    if sleep: time.sleep(sleep)
    return r.json()

def extract_tag_pit(cf_json, taxonomy, tag, unit):
    # Try primary taxonomy/tag
    #print(taxonomy,tag)
    try:
        units = cf_json["facts"][taxonomy][tag]["units"]
        facts = units.get(unit) or units.get(unit.capitalize())
    except KeyError:
        facts = None

    # Fallback to DEI when primary missing
    if not facts and taxonomy != "dei":
        try:
            units = cf_json["facts"]["dei"][tag]["units"]
            facts = units.get(unit) or units.get(unit.capitalize())
        except KeyError:
            facts = None

    if not facts:
        return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])

    df = pd.DataFrame(facts).rename(columns={"end":"period_end","val":"value"})
    for col in ["period_end","filed","start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["source_taxonomy"] = taxonomy
    df["source_tag"] = tag
    df["metric"] = f"{taxonomy}/{tag}"
    df["unit"] = unit
    df["canonical_key"] = None  # default for non-resolved concepts
    return df[["period_end","filed","value","metric","unit","source_taxonomy","source_tag","canonical_key"]].dropna(subset=["filed"])


def dropDuplicateInfo(_df, key_cols):
    '''Drop data when duplicated'''
    df_last = _df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
    return df_last

def resolve_concept(cf_json, canonical_key, compare=False):
    candidates = CANONICAL_CONCEPTS.get(canonical_key, [])
    found = []
    for tax, tag, unit in candidates:
        df = extract_tag_pit(cf_json, tax, tag, unit)
        if not df.empty:
            found.append((tax, tag, unit, df))

    if not found:
        return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])
    # Special logic for Liabilities: compute from components if total missing
    print(candidates, canonical_key)
    if canonical_key == "Liabilities":
        print("im in")
        total_df = next((d for t,g,u,d in found if g=="Liabilities"), None)
        current_df = next((d for t,g,u,d in found if g=="LiabilitiesCurrent"), None)
        noncurrent_df = next((d for t,g,u,d in found if g=="LiabilitiesNoncurrent"), None)
        
        if total_df is None and current_df is not None and noncurrent_df is not None:
            # Merge on period_end and filed, sum the values
            merged = pd.merge(
                current_df, 
                noncurrent_df, 
                on=["period_end", "filed"], how="inner",
                suffixes=("_curr", "_noncurr")
            )
            
            out = pd.DataFrame({
                "period_end": merged["period_end"],
                "filed": merged["filed"],
                "value": merged["value_curr"] + merged["value_noncurr"],
                "metric": "Liabilities",
                "unit": "USD",
                "source_taxonomy": "us-gaap",
                "source_tag": "Liabilities (computed)",
                "canonical_key": canonical_key
            })
            return out
        
        # Otherwise return total if available, or fallback to current
        if total_df is not None:
            out = total_df.copy()
        elif current_df is not None:
            out = current_df.copy()
        else:
            out = noncurrent_df.copy()
        
        out["canonical_key"] = canonical_key
        return out

    if compare and len(found) > 1 and canonical_key == "SharesOutstanding":
        us_df = next((d for t,g,u,d in found if t=="us-gaap"), None)
        dei_df = next((d for t,g,u,d in found if t=="dei"), None)
        if us_df is not None and dei_df is not None:
            uval = us_df.sort_values(["filed","period_end"]).iloc[-1]["value"]
            dval = dei_df.sort_values(["filed","period_end"]).iloc[-1]["value"]
            ratio = max(uval, dval) / min(uval, dval) if min(uval, dval) else float("inf")
            if ratio > 100:
                print("Warning: DQC_0095-like scale mismatch for shares.")
        chosen = us_df if us_df is not None else dei_df
        out = chosen.copy()
        out["canonical_key"] = canonical_key
        return out

    out = found[0][3].copy()
    out["canonical_key"] = canonical_key
    return out

def fetch_facts_latest_for_cik(cik, ticker, targets):
    '''Retrieve each fundamentals for each company, using the SEC data'''
    cf = get_json(f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json")
    series = [extract_tag_pit(cf, *t) for t in targets]
    series.append(resolve_concept(cf, "SharesOutstanding", compare=True))
    series.append(resolve_concept(cf, "Liabilities", compare=True))
    series_nonempty = [df for df in series if not df.empty]
    cols = ["period_end","filed","value","metric","unit","source_taxonomy","source_tag","canonical_key"]
    facts_long = (
        pd.concat(series_nonempty, ignore_index=True)
        if series_nonempty else
        pd.DataFrame(columns=cols)
    )
    facts_latest = dropDuplicateInfo(facts_long, ["metric","unit","period_end"])
    facts_latest["cik"] = cik
    facts_latest["ticker"] = ticker
    return facts_latest


def fundamentals(args):
    '''Retrieve sp500 information from markets.py, and obtain the fundamentals for these'''
    df_all = pd.read_csv(SP500_NAMES_FILE)
    df=test_subset(df_all,args)
    pairs = list(
        df.loc[:, ["CIK", "Symbol"]]
        .assign(CIK=lambda x: x["CIK"].astype(str).str.zfill(10))
        .to_records(index=False)
    )

    universe = [ (cik, sym) for cik, sym in pairs ]
    fundamentals_file= get_fundamental_file(args)
    nstocks = len(universe)
    print(f"Processing {nstocks} stocks")
    all_facts = []
    print(universe)
    for i, (cik, ticker) in enumerate(universe, start=1):
        print(f"{i}/{nstocks}: {ticker}")
        try:
            df_i = fetch_facts_latest_for_cik(cik, ticker, FUNDAMENTAL_VARS)
            all_facts.append(df_i)
        except Exception as e:
            print(f"skip {cik} {ticker}: {e}")

    facts_latest_all = pd.concat(all_facts, ignore_index=True)
    # Sort
    facts_latest_all = facts_latest_all.sort_values(
        ["ticker","metric","unit","period_end","filed"]
    ).reset_index(drop=True)
    # Write to CSV
    print("Saving file to:", fundamentals_file)
    facts_latest_all.to_csv(fundamentals_file, index=False)


