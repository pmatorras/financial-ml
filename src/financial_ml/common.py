from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR /"data"
FIGURE_DIR = ROOT_DIR / "figures"
TEST_DIR = ROOT_DIR / "test"
DEBUG_DIR = ROOT_DIR /"debug"
LOGS_DIR = ROOT_DIR / "logs"
SP500_NAMES_FILE = DATA_DIR/"sp500_list.csv"
SP500_MARKET_FILE = DATA_DIR/"sp500_market.csv"
SP500_MARKET_TEST = DATA_DIR/"sp500_market_test.csv"
SP500_MARKET_DEBUG = DEBUG_DIR/"sp500_market_debug.csv"
SP500_FUNDA_FILE = DATA_DIR/"sp500_fundamentals.csv"
SP500_FUNDA_TEST = DATA_DIR/"sp500_fundamentals_test.csv"
SP500_FUNDA_DEBUG = DEBUG_DIR/"sp500_fundamentals_debug.csv"
PRED_FILE = DATA_DIR/"oof_predictions.csv"
DATA_INTERVAL="1mo"
START_STORE_DATE="1995-01-01"
SP500_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MARKET_KEYS = ["ClosePrice", "r1", "r12", "mom121","vol3","vol12"]
FUNDA_KEYS  = ["BookToMarket","ROE", "ROA", "NetMargin", "Leverage",  'AssetGrowth',  'NetShareIssuance','LogMktCap']

DEBUG_SYMBOLS = ['MRK']# 'CSX'] #['DTE', 'AEP']
'''
FUNDAMENTAL_VARS = [
    ("us-gaap","Assets","USD"),
    ("us-gaap","Liabilities","USD"),
    ("us-gaap","StockholdersEquity","USD"),
    ("us-gaap","Revenues","USD"),
    ("us-gaap","NetIncomeLoss","USD"),
#    ("us-gaap","EarningsPerShareDiluted","USD/share"),
    ("us-gaap","CommonStockSharesOutstanding","shares")
]
'''
CANONICAL_CONCEPTS = {
    "CommonStockSharesOutstanding": [
        ("us-gaap", "CommonStockSharesOutstanding", "shares"),
        ("us-gaap", "CommonStockSharesIssued", "shares"),
        ("dei", "EntityCommonStockSharesOutstanding", "shares"),
        # Add class AB-specific tags
        ("us-gaap", "CommonClassAMember", "shares"),
        ("us-gaap", "CommonClassBMember", "shares"),
        ("us-gaap", "CommonStockClassASharesOutstanding", "shares"),
        ("us-gaap", "CommonStockClassBSharesOutstanding", "shares"),
    ],
    "Assets": [("us-gaap", "Assets", "USD")],
    "Liabilities": [
        ("us-gaap", "Liabilities", "USD"),
        ("us-gaap", "LiabilitiesCurrent", "USD"),  # Current liabilities only
        ("us-gaap", "LiabilitiesNoncurrent", "USD"),  # Non-current only
    ],
    "StockholdersEquity": [("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
                           ("us-gaap", "StockholdersEquity", "USD")
                           ],
    "Revenues": [("us-gaap", "Revenues", "USD"),
                 ("us-gaap", "SalesRevenueNet", "USD"),  
                 ("us-gaap", "RegulatedOperatingRevenue", "USD"),
                 ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
                 ("us-gaap","RevenueFromContractWithCustomerIncludingAssessedTax","USD")],
    "NetIncomeLoss": [
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic", "USD"),
        ("us-gaap", "NetIncomeLoss", "USD"),
        ("us-gaap", "NetIncomeLossAttributableToParent", "USD"),
        ("us-gaap", "ProfitLoss", "USD"),
        ("us-gaap", "IncomeLossFromContinuingOperations", "USD")
    ],    # Might add more
}
def createFolders():
    for p in (DATA_DIR,FIGURE_DIR, LOGS_DIR, TEST_DIR, DEBUG_DIR):
        p.mkdir(parents=True, exist_ok=True)

def get_market_file(args):
    if args.debug: csv_filenm =SP500_MARKET_DEBUG
    elif args.test: csv_filenm = SP500_MARKET_TEST
    else: csv_filenm = SP500_MARKET_FILE
    print("opening", csv_filenm)
    return csv_filenm

def get_fundamental_file(args):
    if args.debug: csv_filenm =SP500_FUNDA_DEBUG
    elif args.test: csv_filenm = SP500_FUNDA_TEST
    else: csv_filenm = SP500_FUNDA_FILE
    print("opening", csv_filenm)
    return csv_filenm