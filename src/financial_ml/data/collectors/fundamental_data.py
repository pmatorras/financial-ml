import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from financial_ml.utils.config import SP500_NAMES_FILE, CANONICAL_CONCEPTS, CIK_OVERRIDES
from financial_ml.utils.paths import get_fundamental_file
from financial_ml.data.collectors.utils import filter_market_subset

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
            if equity_df is None:
                equity_df = df
            else:
                # Append and remove duplicates
                equity_df = pd.concat([equity_df, df], ignore_index=True)
                equity_df = equity_df.drop_duplicates(subset=["period_end", "filed"], keep="first")

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

def resolve_concept_real(cf_json, canonical_key, args, compare=False, merge=False):
    candidates = CANONICAL_CONCEPTS.get(canonical_key, [])
    found = []
    
    for tax, tag, unit in candidates:
        df = extract_tag_pit(cf_json, tax, tag, unit)
        if not df.empty:
            found.append((tax, tag, unit, df))

    
    if args.debug: print("canon", canonical_key)
    
    # Special case: CommonStockSharesOutstanding
    if canonical_key == "CommonStockSharesOutstanding":
        # Try to find class-specific shares
        classA_df = next((d for t,g,u,d in found if "ClassA" in g), None)
        classB_df = next((d for t,g,u,d in found if "ClassB" in g), None)
        
        # If both classes exist, sum them
        if classA_df is not None and classB_df is not None:
            merged = pd.merge(
                classA_df, classB_df,
                on=["period_end", "filed"], 
                how="outer",
                suffixes=("_A", "_B")
            )
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
            
            # Fill gaps with non-class tags
            existing_dates = set(out['period_end'])
            for t, g, u, df in found:
                if 'ClassA' not in g and 'ClassB' not in g:
                    new_data = df[~df['period_end'].isin(existing_dates)].copy()
                    if not new_data.empty:
                        new_data['canonical_key'] = canonical_key
                        out = pd.concat([out, new_data], ignore_index=True)
            
            return out.sort_values('period_end').reset_index(drop=True)
        
        # Priority order (FIXED - WeightedAverage before Entity)
        priority_tags = [
            "WeightedAverageNumberOfDilutedSharesOutstanding", 
            "WeightedAverageNumberOfSharesOutstandingBasic",  
            "CommonStockSharesOutstanding",
            "CommonStockSharesIssued",
            "EntityCommonStockSharesOutstanding" 
        ]
        
        # Collect ALL available tags
        tag_dfs = {}
        for priority_tag in priority_tags:
            match = next((df for t, g, u, df in found if g == priority_tag), None)
            if match is not None:
                tag_dfs[priority_tag] = match.copy()
        
        if not tag_dfs:
            return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])
        
        # Start with highest priority available tag
        first_tag = next((t for t in priority_tags if t in tag_dfs), None)
        out = tag_dfs[first_tag].copy()
        
        # Fill gaps with ALL remaining tags
        existing_dates = set(out['period_end'])
        for priority_tag in priority_tags:
            if priority_tag == first_tag:
                continue  # Skip the one we started with
            
            if priority_tag in tag_dfs:
                new_data = tag_dfs[priority_tag][~tag_dfs[priority_tag]['period_end'].isin(existing_dates)].copy()
                if not new_data.empty:
                    out = pd.concat([out, new_data], ignore_index=True)
                    existing_dates.update(new_data['period_end'])
                    
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
    if not found:
        return pd.DataFrame(columns=["period_end","filed","value","metric","unit"])
    # Merge logic for other concepts
    if merge and len(found) > 1:
        all_dfs = [df.copy() for _, _, _, df in found]
        combined = pd.concat(all_dfs, ignore_index=True)
        
        combined = combined.sort_values(
            ["period_end", "filed"], 
            ascending=[True, False]
        ).drop_duplicates(
            subset=["period_end"], 
            keep="first"
        ).sort_values("period_end")
        
        combined["metric"] = canonical_key
        combined["canonical_key"] = canonical_key
        
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
    
    # Default: return first found, but TRY TO FILL GAPS
    out = found[0][3].copy()
    existing_dates = set(out['period_end'])
    
    for _, _, _, df in found[1:]:
        new_data = df[~df['period_end'].isin(existing_dates)].copy()
        if not new_data.empty:
            out = pd.concat([out, new_data], ignore_index=True)
            existing_dates.update(new_data['period_end'])
    
    out["canonical_key"] = canonical_key
    return out.sort_values('period_end').reset_index(drop=True)




def fetch_facts_latest_for_cik(cik, ticker, dict_facts, args):
    """
    Retrieve facts for one company, with automatic fallback to alternate CIKs.
    """
    # Check if this ticker has multiple CIKs
    if ticker in CIK_OVERRIDES:
        cik_list = CIK_OVERRIDES[ticker]
    else:
        cik_list = [cik]
    
    all_series = []
    
    for try_cik in cik_list:
        try:
            cf = get_json(f"{BASE}/api/xbrl/companyfacts/CIK{try_cik}.json")
            
            requested_keys = list(dict_facts.keys())
            
            for canon_key in requested_keys:
                needs_merge = canon_key in ["Revenues", "CommonStockSharesOutstanding", "NetIncomeLoss", "Liabilities"]
                needs_compare = canon_key == "CommonStockSharesOutstanding"
                
                if args.debug:
                    print(f"  pre concept: {canon_key} from CIK {try_cik}")
                
                dfk = resolve_concept_real(cf, canon_key, args, compare=needs_compare, merge=needs_merge)
                
                if not dfk.empty:
                    dfk['source_cik'] = try_cik
                    dfk['canonicalkey'] = canon_key  # ADD THIS LINE
                    all_series.append(dfk)
        
        except Exception as e:
            if args.debug:
                print(f"  CIK {try_cik} failed: {e}")
            continue
    
    if not all_series:
        # Return empty DataFrame
        cols = ['period_end','filed','value','metric','unit','sourcetaxonomy','sourcetag','canonicalkey']
        return pd.DataFrame(columns=cols + ['cik', 'ticker'])
    
    # Combine data from all CIKs
    facts_long = pd.concat(all_series, ignore_index=True)
    
    # NOW drop duplicates - canonicalkey exists
    facts_latest = dropDuplicateInfo(facts_long, ['canonicalkey','unit','period_end','filed'])
    
    facts_latest['cik'] = ','.join(cik_list)
    facts_latest['ticker'] = ticker
    
    return facts_latest



def collect_fundamentals_data(args):
    '''Retrieve sp500 information from markets.py, and obtain the fundamentals for these'''
    df_all = pd.read_csv(SP500_NAMES_FILE)
    df=filter_market_subset(df_all,args)
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


