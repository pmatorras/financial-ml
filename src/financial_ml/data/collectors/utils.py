'''
Utility functions comon for the data collectors
'''
from financial_ml.utils.config import DEBUG_SYMBOLS
def filter_market_subset(df, args):
    '''
    Filter SP500 based on the run mode (debug,test or production).
    Returns a skimmed df
    '''
    if args.debug:
        if args.ticker:
            debug_symbols=args.ticker
        else:
            debug_symbols = DEBUG_SYMBOLS
        print(f"returning the following symbols :{debug_symbols}")
        return(df[df['Symbol'].isin(debug_symbols)])
    elif args.test: 
        print("returning only test subset")
        return df.sort_values('Date added').head(50)
    else:
        return df