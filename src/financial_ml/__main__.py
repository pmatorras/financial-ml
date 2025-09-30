import argparse
from .markets import store_info
from .train import train
from .fundamentals import fundamentals
def main():
    parser = argparse.ArgumentParser(description="Compare GDP and Inflation for selected countries")
    parser.add_argument("-nt", "--newtable", action="store_true", help="Update sp500 table")    
    parser.add_argument("-ni", "--newinfo", action="store_true", help="Update sp500 financial information") 
    parser.add_argument("--test"   , action="store_true", help="Test on a smaller subset of 50")    
    parser.add_argument("--train"   , action="store_true", help="perform the training")
    parser.add_argument("-f", "--fundamentals"   , action="store_true", help="Download fundamentals")   
   
    args = parser.parse_args()
    if args.newinfo or args.newtable: store_info(args)
    if args.train: train(args)
    if args.fundamentals: fundamentals(args)



if __name__ == "__main__":
    main()