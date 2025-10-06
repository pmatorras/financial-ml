from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR /"data"
FIGURE_DIR = ROOT_DIR / "figures"
TEST_DIR = ROOT_DIR / "test"
LOGS_DIR = ROOT_DIR / "logs"
SP500_NAMES_FILE = DATA_DIR/"sp500_list.csv"
SP500_MARKET_FILE = DATA_DIR/"sp500_market.csv"
SP500_MARKET_TEST = DATA_DIR/"sp500_market_test.csv"
SP500_FUNDA_FILE = DATA_DIR/"sp500_fundamentals.csv"
SP500_FUNDA_TEST = DATA_DIR/"sp500_fundamentals_test.csv"

DATA_INTERVAL="1mo"
START_STORE_DATE="1995-01-01"
SP500_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MARKET_KEYS = ["ClosePrice", "r1", "r12", "mom121","vol3","vol12"]
FUNDA_KEYS  = ["BookToMarket","ROE", "ROA", "NetMargin", "Leverage",  'AssetGrowth',  'NetShareIssuance','LogMktCap']

FUNDAMENTAL_VARS = [
    ("us-gaap","Assets","USD"),
    ("us-gaap","Liabilities","USD"),
    ("us-gaap","StockholdersEquity","USD"),
    ("us-gaap","Revenues","USD"),
    ("us-gaap","NetIncomeLoss","USD"),
#    ("us-gaap","EarningsPerShareDiluted","USD/share"),
    ("us-gaap","CommonStockSharesOutstanding","shares"), #Careful with double counting this (see https://www.perplexity.ai/search/i-am-developing-a-framework-th-6QC.Sc1JS1GjJQBfELScNw#5)
]
CANONICAL_CONCEPTS = {
    "SharesOutstanding": [
        ("us-gaap", "CommonStockSharesOutstanding", "shares"),
        ("dei", "EntityCommonStockSharesOutstanding", "shares"),
    ],
    "Assets": [("us-gaap", "Assets", "USD")],
    "Liabilities": [("us-gaap", "Liabilities", "USD")],
    "Equity": [("us-gaap", "StockholdersEquity", "USD")],
    "Revenue": [("us-gaap", "Revenues", "USD")],
    "NetIncome": [("us-gaap", "NetIncomeLoss", "USD")],
    # Might add more
}
def createFolders():
    for p in (DATA_DIR,FIGURE_DIR, LOGS_DIR, TEST_DIR):
        p.mkdir(parents=True, exist_ok=True)

