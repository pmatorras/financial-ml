import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from financial_ml.utils.config import SP500_NAMES_FILE, CANONICAL_CONCEPTS
from financial_ml.utils.paths import get_fundamental_file
from financial_ml.markets import test_subset

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
    #print(f"Available tags for {tag}:", list(cf_json['facts'].get('us-gaap', {}).keys()))

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
    '''Drop data when duplicated,, keeping latest filed date'''
     # Identify the grouping columns (exclude 'filed' if it's in key_cols)
    group_cols = [col for col in key_cols if col != "filed"]
    
    # Sort by group columns + filed descending to get latest filing first
    df_sorted = _df.sort_values(
        group_cols + ["filed"], 
        ascending=[True] * len(group_cols) + [False]
    )
    
    # Keep first (= latest filed) for each group
    df_dedup = df_sorted.drop_duplicates(subset=group_cols, keep="first")
    
    # Re-sort for clean output
    df_last = df_dedup.sort_values(group_cols).reset_index(drop=True)
    
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
    if canonical_key == "Liabilities":
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

def _compute_liabilities_from_components(current_df, noncurrent_df, min_coverage=0.80):
    """
    Compute liabilities from current + noncurrent with coverage check.
    Returns None if coverage is insufficient.
    """
    if current_df is None or noncurrent_df is None:
        return None
    
    overlap = pd.merge(
        current_df[["period_end", "filed"]],
        noncurrent_df[["period_end", "filed"]],
        on=["period_end", "filed"],
        how="inner"
    )
    
    if len(overlap) == 0:
        return None
    
    coverage_current = len(overlap) / len(current_df)
    coverage_noncurrent = len(overlap) / len(noncurrent_df)
    
    # Quality gate
    if coverage_current < min_coverage or coverage_noncurrent < min_coverage:
        return None
    
    # Safe to compute
    merged = pd.merge(
        current_df, noncurrent_df,
        on=["period_end", "filed"],
        how="inner",
        suffixes=("_curr", "_noncurr")
    )
    
    return pd.DataFrame({
        "period_end": merged["period_end"],
        "filed": merged["filed"],
        "value": merged["value_curr"] + merged["value_noncurr"],
        "metric": "Liabilities",
        "unit": "USD",
        "source_taxonomy": "us-gaap",
        "source_tag": f"Liabilities (computed, cov={coverage_current:.1%},{coverage_noncurrent:.1%})",
    })


def _compute_liabilities_from_balance_sheet(cf_json):
    """Compute liabilities from Assets - Equity."""
    assets_candidates = CANONICAL_CONCEPTS.get("Assets", [])
    equity_candidates = CANONICAL_CONCEPTS.get("StockholdersEquity", [])
    
    assets_df = None
    for tax, tag, unit in assets_candidates:
        df = extract_tag_pit(cf_json, tax, tag, unit)
        if not df.empty:
            assets_df = df
            break
    
    equity_df = None
    for tax, tag, unit in equity_candidates:
        df = extract_tag_pit(cf_json, tax, tag, unit)
        if not df.empty:
            equity_df = df
            break
    
    if assets_df is None or equity_df is None:
        return None
    
    merged = pd.merge(
        assets_df, equity_df,
        on=["period_end", "filed"],
        how="inner",
        suffixes=("_assets", "_equity")
    )
    
    if len(merged) == 0:
        return None
    
    return pd.DataFrame({
        "period_end": merged["period_end"],
        "filed": merged["filed"],
        "value": merged["value_assets"] - merged["value_equity"],
        "metric": "Liabilities",
        "unit": "USD",
        "source_taxonomy": "us-gaap",
        "source_tag": "Liabilities (from Assets-Equity)",
    })




def resolve_concept2(cf_json, canonical_key, args, compare=False, merge=False):
    candidates = CANONICAL_CONCEPTS.get(canonical_key, [])
    found = []
    
    for tax, tag, unit in candidates:
        df = extract_tag_pit(cf_json, tax, tag, unit)
        if not df.empty:
            #print("candidate not empty")#, tag, "\n---\n", df)
            found.append((tax, tag, unit, df))

    if not found:
        return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])
    # Special logic for CommonStockSharesOutstanding: sum Class A + Class B
    if args.debug: print("canon", canonical_key)
    if canonical_key == "CommonStockSharesOutstanding":
        #print("im in")
        # Try to find class-specific shares
        classA_df = next((d for t,g,u,d in found if "ClassA" in g), None)
        classB_df = next((d for t,g,u,d in found if "ClassB" in g), None)
        #print("classes", classA_df, classB_df)
        # If both classes exist, sum them
        if classA_df is not None and classB_df is not None:
            merged = pd.merge(
                classA_df,
                classB_df,
                on=["period_end", "filed"], 
                how="outer",
                suffixes=("_A", "_B")
            )
            # Fill NaN with 0 for outer join, then sum
            merged["value_A"] = merged["value_A"].fillna(0)
            merged["value_B"] = merged["value_B"].fillna(0)
            
            out = pd.DataFrame({
                "period_end": merged["period_end"],
                "filed": merged["filed"],
                "value": merged["value_A"] + merged["value_B"],
                "metric": "CommonStockSharesOutstanding",
                "unit": "shares",
                "source_taxonomy": "us-gaap",
                "source_tag": "CommonStockSharesOutstanding (Class A + B)",
                "canonical_key": canonical_key
            })
            return out
        
        # Otherwise use standard total if available
        priority_tags = [
            "CommonStockSharesOutstanding",      # Primary
            "CommonStockSharesIssued",           # Fallback 1
            "EntityCommonStockSharesOutstanding" # Fallback 2 (dei)
        ]
        
        # Collect dataframes for each priority tag
        tag_dfs = {}
        for priority_tag in priority_tags:
            match = next((df for t, g, u, df in found if g == priority_tag), None)
            if match is not None:
                tag_dfs[priority_tag] = match.copy()
        
        # Start with highest priority tag
        if "CommonStockSharesOutstanding" in tag_dfs:
            out = tag_dfs["CommonStockSharesOutstanding"].copy()
        elif "CommonStockSharesIssued" in tag_dfs:
            out = tag_dfs["CommonStockSharesIssued"].copy()
        elif "EntityCommonStockSharesOutstanding" in tag_dfs:
            out = tag_dfs["EntityCommonStockSharesOutstanding"].copy()
        else:
            return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])
        
        # Fill gaps with lower priority tags
        for priority_tag in priority_tags[1:]:  # Skip the first one we already used
            if priority_tag in tag_dfs:
                fallback = tag_dfs[priority_tag]
                # Find dates in fallback that don't exist in out
                existing_dates = set(out['period_end'])
                new_data = fallback[~fallback['period_end'].isin(existing_dates)]
                if not new_data.empty:
                    out = pd.concat([out, new_data], ignore_index=True)
                    
        out['canonical_key'] = canonical_key
        return out.sort_values('period_end').reset_index(drop=True)
    
    if canonical_key == "Liabilities":
            total_df = next((d for t,g,u,d in found if g=="Liabilities"), None)
            current_df = next((d for t,g,u,d in found if g=="LiabilitiesCurrent"), None)
            noncurrent_df = next((d for t,g,u,d in found if g=="LiabilitiesNoncurrent"), None)
            
            # Start with empty result
            result = None
            
            # Layer 1: Fill with direct Liabilities
            if total_df is not None:
                result = total_df.copy()
                result["source_tag"] = "Liabilities (direct)"
            
            # Layer 2: Fill gaps with component sum (coverage-checked)
            computed_components = _compute_liabilities_from_components(current_df, noncurrent_df)
            if computed_components is not None:
                if result is None:
                    result = computed_components
                else:
                    # Add only NEW periods
                    existing_periods = set(zip(result["period_end"], result["filed"]))
                    new_periods = computed_components[
                        ~computed_components.apply(
                            lambda row: (row["period_end"], row["filed"]) in existing_periods,
                            axis=1
                        )
                    ]
                    if len(new_periods) > 0:
                        result = pd.concat([result, new_periods], ignore_index=True)
            
            # Layer 3: Fill remaining gaps with balance sheet equation
            computed_balance = _compute_liabilities_from_balance_sheet(cf_json)
            if computed_balance is not None:
                if result is None:
                    result = computed_balance
                else:
                    # Add only NEW periods
                    existing_periods = set(zip(result["period_end"], result["filed"]))
                    new_periods = computed_balance[
                        ~computed_balance.apply(
                            lambda row: (row["period_end"], row["filed"]) in existing_periods,
                            axis=1
                        )
                    ]
                    if len(new_periods) > 0:
                        result = pd.concat([result, new_periods], ignore_index=True)
            
            # Return whatever we got (or fail cleanly)
            if result is not None and len(result) > 0:
                result["canonical_key"] = canonical_key
                result = result.sort_values(["period_end", "filed"]).reset_index(drop=True)
                return result
            
            return pd.DataFrame(columns=["period_end","filed","value","metric","unit","canonical_key"])


    '''Old logic
    # Special logic for Liabilities: compute from components if total missing
    if canonical_key == "Liabilities":
        print("im in liabilities")
        total_df = next((d for t,g,u,d in found if g=="Liabilities"), None)
        current_df = next((d for t,g,u,d in found if g=="LiabilitiesCurrent"), None)
        noncurrent_df = next((d for t,g,u,d in found if g=="LiabilitiesNoncurrent"), None)
        
        if total_df is None and current_df is not None and noncurrent_df is not None:
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
        return pd.DataFrame(columns=["period_end","filed","value","metric","unit","canonical_key"])
        '''
    
    # Merge logic for concepts with multiple tags covering different periods
    if merge and len(found) > 1:
        all_dfs = [df.copy() for _, _, _, df in found]
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates, preferring later filings
        combined = combined.sort_values(
            ["period_end", "filed"], 
            ascending=[True, False]
        ).drop_duplicates(
            subset=["period_end"], 
            keep="first"
        ).sort_values("period_end")
        
        combined["metric"] = canonical_key
        combined["canonical_key"] = canonical_key
        
        # Special comparison for SharesOutstanding
        if compare:
            us_df = next((df for t, _, _, df in found if t == "us-gaap"), None)
            dei_df = next((df for t, _, _, df in found if t == "dei"), None)
            
            if us_df is not None and dei_df is not None:
                uval = us_df.sort_values(["filed","period_end"]).iloc[-1]["value"]
                dval = dei_df.sort_values(["filed","period_end"]).iloc[-1]["value"]
                ratio = max(uval, dval) / min(uval, dval) if min(uval, dval) else float("inf")
                if ratio > 100:
                    print("Warning: DQC_0095-like scale mismatch for shares.")
        
        return combined
    
    # Default: return first found
    out = found[0][3].copy()
    out["canonical_key"] = canonical_key
    return out

def fetch_facts_latest_for_cik(cik, ticker, dict_facts,args):
    """
    Retrieve facts for one company, resolving to canonical_key for all requested concepts.
    """
    cf = get_json(f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json")
    requested_keys = list(dict_facts.keys())

    # Resolve each canonical key once
    series = []
    for canon_key in requested_keys:
        needs_merge = canon_key in ["Revenues", "CommonStockSharesOutstanding", "NetIncomeLoss"]
        needs_compare = canon_key == "CommonStockSharesOutstanding"
        if args.debug: print("pre concept", canon_key)
        df_k = resolve_concept2(cf, canon_key,args, compare=needs_compare, merge=needs_merge)

        #df_k = resolve_concept(cf, canon_key, compare=(canon_key == "SharesOutstanding"))
        if not df_k.empty:
            # resolve_concept already sets the canonical_key column
            series.append(df_k)
    # Concatenate and deduplicate by (canonical_key, unit, period_end)
    cols = ["period_end","filed","value","metric","unit","source_taxonomy","source_tag","canonical_key"]
    facts_long = (pd.concat(series, ignore_index=True) if series else pd.DataFrame(columns=cols))

    # Prefer the latest filing per canonical concept/date
    facts_latest = dropDuplicateInfo(facts_long, ["canonical_key","unit","period_end","filed"])
    #facts_latest = facts_long
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
    for i, (cik, ticker) in enumerate(universe, start=1):
        print(f"{i}/{nstocks}: {ticker}")
        try:
            df_i = fetch_facts_latest_for_cik(cik, ticker, CANONICAL_CONCEPTS, args)
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


