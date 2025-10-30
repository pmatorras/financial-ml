"""Main entry point for financial_ml CLI."""
from financial_ml.cli.parser import cli
from financial_ml.data.collectors import collect_market_data, collect_fundamentals_data, collect_sentiment_data
from financial_ml.models import train
from financial_ml.utils.paths import create_parent_folders
from financial_ml.portfolio import run_backtest
from financial_ml.evaluation.analyze import analyze_models

def main(argv=None):
    parser = cli()
    args = parser.parse_args(argv)
    args.best_model = 'rf_cal'
    print(f"Running the code, with arguments: {args}")
    create_parent_folders()


    if args.cmd == "market":
        print("Downloading market data...")
        collect_market_data(args)
    elif args.cmd == "fundamentals":
        print("Downloading fundamentals")
        collect_fundamentals_data(args)
    elif args.cmd == "sentiment":
        print("Downloading fundamentals")
        collect_sentiment_data(args)
    elif args.cmd == "train":
        print("Performing training")
        train(args)
    elif args.cmd == "analyze":
        print("Analyzing models")
        analyze_models(args)
    elif args.cmd == "portfolio":
        print(f"Getting a {args.type} type portfolio")
        run_backtest(args, args.pertop, args.perbot)
    return 0
