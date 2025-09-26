from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR /"data"
FIGURE_DIR = ROOT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"
SP500_NAMES_FILE = DATA_DIR/"sp500_list.csv"
SP500_MARKET_FILE = DATA_DIR/"sp500_values.csv"
SP500_MARKET_TEST = DATA_DIR/"sp500_test.csv"

for p in (DATA_DIR,FIGURE_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)