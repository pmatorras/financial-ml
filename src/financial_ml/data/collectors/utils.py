'''
Utility functions comon for the data collectors
'''
from financial_ml.utils.config import DEBUG_SYMBOLS
def test_subset(df, args):
    '''
    Filter SP500 based on the run mode (debug,test or production).
    Returns a skimmed df
    '''
    if args.debug:
        print(f"returning the following symbols :{DEBUG_SYMBOLS}")
        return(df[df['Symbol'].isin(DEBUG_SYMBOLS)])
    elif args.test: 
        print("returning only test subset")
        return df.sort_values('Date added').head(50)
    else:
        return df