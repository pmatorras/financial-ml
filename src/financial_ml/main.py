"""Main entry point for financial_ml CLI."""
from financial_ml.cli.parser import cli
from financial_ml.data.collectors import collect_market_data, collect_fundamentals
from financial_ml.models import train
from financial_ml.utils.paths import createFolders
from financial_ml.portfolio import run_backtest
from financial_ml.evaluation.analyze import analyze_models

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
        print(f"Getting a {args.type} type portfolio")
        exit()
        run_backtest(args, args.pertop, args.perbot)
    return 0
