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

#number of folds
N_SPLITS = 3

#Display formatting constants
SEPARATOR_WIDTH = 70


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

DEBUG_SYMBOLS = ['DPZ'] #'BRK.B', # 'CSX'] #['DTE', 'AEP']
UNFIXABLE = ['ERIE', 'PSKY', 'STZ', 'TKO', 'V'] #Check issues page (https://github.com/pmatorras/financial-ml/issues) for more info
CIK_OVERRIDES = {
    'BLK': ['0001364742', '0002012383'],  # Old CIK first for historical data
    'APA': ['0000006769', '0001841666'],  # Apache Corp (old) → APA Corp (new, March 2021)

    # Add more as needed
}

CANONICAL_CONCEPTS = {
    "CommonStockSharesOutstanding": [
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares"), 
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
        # General/agregate
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),  
        ("us-gaap", "RegulatedOperatingRevenue", "USD"),
        ("us-gaap", "RevenuesNetOfInterestExpense", "USD"), 
        # Post 2018 (ASC 606)
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap","RevenueFromContractWithCustomerIncludingAssessedTax","USD"),
        # For financial institutions
        ("us-gaap", "InterestAndFeeIncomeLoansAndLeases", "USD"),  # Financial services
        ("us-gaap", "InterestIncomeOperating", "USD"),  # Banks/lenders
        # Pre-2018 (specific/narrow) - use as last resort
        ("us-gaap", "SalesRevenueGoodsNet", "USD"),  # Goods only (may miss service revenue)
        ("us-gaap", "SalesRevenueServicesNet", "USD"),  # Services only (may miss goods revenue)
     
    ],
    "NetIncomeLoss": [
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic", "USD"),
        ("us-gaap", "NetIncomeLoss", "USD"),
        ("us-gaap", "NetIncomeLossAttributableToParent", "USD"),
        ("us-gaap", "ProfitLoss", "USD"),
        ("us-gaap", "IncomeLossFromContinuingOperations", "USD")
    ],   
}

