from pathlib import Path

#Directory paths
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR /"data"
FIGURE_DIR = ROOT_DIR / "figures"
TEST_DIR = ROOT_DIR / "test"
DEBUG_DIR = ROOT_DIR /"debug"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"


#SP500 company info list
SP500_NAMES_FILE = DATA_DIR / "sp500_list.csv"

#Market data files
SP500_MARKET_FILE = DATA_DIR / "sp500_market.csv"
SP500_MARKET_TEST = DATA_DIR / "sp500_market_test.csv"
SP500_MARKET_DEBUG = DEBUG_DIR / "sp500_market_debug.csv"

#Fundamentals data files
SP500_FUNDA_FILE = DATA_DIR / "sp500_fundamentals.csv"
SP500_FUNDA_TEST = DATA_DIR / "sp500_fundamentals_test.csv"
SP500_FUNDA_DEBUG = DEBUG_DIR / "sp500_fundamentals_debug.csv"

#Prediction files
SP500_PRED_FILE = DATA_DIR / "sp500_oof_predictions.csv"
SP500_PRED_TEST = DATA_DIR / "sp500_oof_predictions_test.csv"
SP500_PRED_DEBUG = DEBUG_DIR / "sp500_oof_predictions_debug.csv"


DATA_INTERVAL="1mo"
START_STORE_DATE="1995-01-01"
SP500_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MARKET_KEYS = [
    "ClosePrice", 
    "r1", 
    "r12", 
    "mom121",
    "vol3",
    "vol12"]
FUNDA_KEYS  = [
    "BookToMarket",
    "ROE",
    "ROA", 
    "NetMargin", 
    "Leverage",  
    'AssetGrowth',  
    'NetShareIssuance',
    'LogMktCap'
    ]

DEBUG_SYMBOLS = ['BRK.B', 'BLK']# 'CSX'] #['DTE', 'AEP']

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
    "StockholdersEquity": [
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
        ("us-gaap", "StockholdersEquity", "USD")
    ],
    "Revenues": [
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),  
        ("us-gaap", "RegulatedOperatingRevenue", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap","RevenueFromContractWithCustomerIncludingAssessedTax","USD")
    ],
    "NetIncomeLoss": [
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic", "USD"),
        ("us-gaap", "NetIncomeLoss", "USD"),
        ("us-gaap", "NetIncomeLossAttributableToParent", "USD"),
        ("us-gaap", "ProfitLoss", "USD"),
        ("us-gaap", "IncomeLossFromContinuingOperations", "USD")
    ],   
}

