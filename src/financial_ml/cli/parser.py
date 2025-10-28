"""Command-line argument parser setup."""
import argparse
from financial_ml.cli.validators import (
    list_from_string,
    max_features_type,
    percentage,
    validate_max_samples
    )


def cli():
    parser = argparse.ArgumentParser(prog="financial_ml",
                                     description="S&P 500 data pipeline: fetch, fundamentals, train")
    parser.add_argument("--test", action="store_true", help="Run on test subset (≈50)")
    parser.add_argument("-d", "--debug", action="store_true", help="Verbose debug logging")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("market", help="Download/refresh market data")
    p_info.add_argument("--newtable","-nt", action="store_true", help="Refresh S&P 500 constituents")
    p_info.add_argument("--newinfo", "-ni", action="store_true", help="Refresh historical market data")
    p_sentiment = sub.add_parser("sentiment", help="Download/refresh sentiment data")

    p_funda = sub.add_parser("fundamentals", help="Download/refresh fundamentals")

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument('--tree-depth', help="Tree depth", type=int, default=3, choices=[3, 4, 5])
    p_train.add_argument('--tree-nestimators', help="number of trees", type=int, default=50, choices=[50, 100, 200])
    p_train.add_argument('--tree-max-features', help="maximum features in the tree", type=max_features_type, default='log2')
    p_train.add_argument('--tree-max-samples', help="Fraction of samples to train each tree (0.0-1.0), or None for all", type=validate_max_samples, default=None)

    p_train.add_argument('--trim-mode', help="Do we want a trimmed version of the fundamentals?", type=str, default='trim', choices= ["trim", "all"])
    p_train.add_argument("--vix-interactions", "-vi", action="store_true", help="Calculate vix interactions")

    p_train.add_argument("-s", "--save", action="store_true", help="Save models")


    p_anal = sub.add_parser("analyze", help="Analize models")

    p_portfolio = sub.add_parser("portfolio", help="Create portfolio")
    p_portfolio.add_argument('--type', help="which type of portfolio to build", type=str, default='100long', choices= ["100long", "longshort", "130-30"])
    p_portfolio.add_argument('--pertop', help="Percentage top portfolio used", type=percentage, default=10)
    p_portfolio.add_argument('--perbot', help="Percentage bottom portfolio used", type=percentage, default=10)

    for sp in (p_info, p_sentiment, p_funda, p_anal, p_train, p_portfolio):
        sp.add_argument("--test", action="store_true", help="Run on test subset (~50)")
        sp.add_argument("-d", "--debug", action="store_true", help="Verbose debug logging")
        sp.add_argument("-v", "--verbose", action="store_true", help="Verbose additional info without doing a small debug subset")
        sp.add_argument("--only-market", dest="only_market",
                         help="Explicitly don't include fundamentals in training features", action="store_true")
        sp.add_argument("--do-sentiment", dest="do_sentiment",
                         help="Explicitly include sentimental data in training features", action="store_true")      
        sp.add_argument('--use-enhanced', action='store_true', help='Include enhanced features (ranks, interactions, reversal)')
        sp.add_argument("--ticker", help="chose ml to display", type=list_from_string, default=None)
        sp.add_argument("--model", "-m", help="chose ml to display", type=str, default='all', choices= ["all", "logreg_l1", "logreg_l2", "rf", "rf_cal", "gb", "ensemble"])

    return parser