import argparse
from .markets import store_info
from .train import train
from .fundamentals import fundamentals
from .common import createFolders
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
    # add funda-specific options if any

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument("--use-fundamentals", action="store_true",
                         help="Include fundamentals in training features")
    #ensure --test --debug can go anywhere
    for sp in (p_info, p_funda, p_train):
        sp.add_argument("--test", action="store_true", help="Run on test subset (≈50)")
        sp.add_argument("-d", "--debug", action="store_true", help="Verbose debug logging")


    return parser

def main(argv=None):
    parser = cli()
    args = parser.parse_args(argv)

    print(f"Running the code, with arguments: {args}")
    createFolders()


    if args.cmd == "market":
        print("Downloading market data...")
        store_info(args)
    elif args.cmd == "fundamentals":
        print("Downloading fundamentals")
        fundamentals(args)
    elif args.cmd == "train":
        print("Performing training")
        train(args)
    return 0