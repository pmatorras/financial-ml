import argparse
from financial_ml.data.collectors import collect_market_data, collect_fundamentals
from financial_ml.models import train
from financial_ml.utils.paths import createFolders
from financial_ml.portfolio import run_backtest
from financial_ml.evaluation.analyze import analyze_models
def cli():
    parser = argparse.ArgumentParser(prog="financial_ml",
                                     description="S&P 500 data pipeline: fetch, fundamentals, train")
    parser.add_argument("--test", action="store_true", help="Run on test subset (≈50)")
    parser.add_argument("-d", "--debug", action="store_true", help="Verbose debug logging")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("market", help="Download/refresh market data")
    p_info.add_argument("--newtable","-nt", action="store_true", help="Refresh S&P 500 constituents")
    p_info.add_argument("--newinfo", "-ni", action="store_true", help="Refresh historical market data")

    p_funda = sub.add_parser("fundamentals", help="Download/refresh fundamentals")

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument('--use-enhanced', action='store_true',
                   help='Include enhanced features (ranks, interactions, reversal)')

    p_anal = sub.add_parser("analyze", help="Analize models")

    p_portfolio = sub.add_parser("portfolio", help="Create portfolio")


    #ensure --test --debug can go anywhere
    for sp in (p_info, p_funda, p_anal, p_train, p_portfolio):
        sp.add_argument("--test", action="store_true", help="Run on test subset (≈50)")
        sp.add_argument("-d", "--debug", action="store_true", help="Verbose debug logging")
        sp.add_argument("--only-market", dest="only_market",
                         help="Explicitly don't include fundamentals in training features", action="store_true")
        sp.add_argument("--model", "-m", help="chose ml to display", type=str, default='all', choices= ["all", "logreg_l1", "logreg_l2", "rf", "rf_cal", "gb"])

    return parser

def main(argv=None):
    parser = cli()
    args = parser.parse_args(argv)
    args.best_model = 'rf_cal'
    print(f"Running the code, with arguments: {args}")
    createFolders()


    if args.cmd == "market":
        print("Downloading market data...")
        collect_market_data(args)
    elif args.cmd == "fundamentals":
        print("Downloading fundamentals")
        collect_fundamentals(args)
    elif args.cmd == "train":
        print("Performing training")
        train(args)
    elif args.cmd == "analyze":
        print("Analyzing models")
        analyze_models(args)
    elif args.cmd == "portfolio":
        print("Getting portfolio")
        run_backtest(args)
    return 0
