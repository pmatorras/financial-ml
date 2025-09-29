from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR /"data"
FIGURE_DIR = ROOT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"
SP500_NAMES_FILE = DATA_DIR/"sp500_list.csv"
SP500_MARKET_FILE = DATA_DIR/"sp500_values.csv"
SP500_MARKET_TEST = DATA_DIR/"sp500_values_test.csv"
SP500_FUNDA_FILE = DATA_DIR/"sp500_fundamentals.csv"
SP500_FUNDA_TEST = DATA_DIR/"sp500_fundamentals_test.csv"

DATA_INTERVAL="1mo"
START_STORE_DATE="1995-01-01"
SP500_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

for p in (DATA_DIR,FIGURE_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)