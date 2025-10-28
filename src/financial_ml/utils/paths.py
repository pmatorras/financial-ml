from pathlib import Path
from typing import Optional
from financial_ml.utils.config import (
    DATA_DIR, FIGURE_DIR, LOGS_DIR, TEST_DIR, DEBUG_DIR, MODELS_DIR,
    SP500_MARKET_FILE, SP500_MARKET_TEST, SP500_MARKET_DEBUG,
    SP500_FUNDA_FILE, SP500_FUNDA_TEST, SP500_FUNDA_DEBUG,
)

def create_parent_folders():
    """Create all required project directories if they don't exist."""
    for p in (DATA_DIR,FIGURE_DIR, LOGS_DIR, TEST_DIR, DEBUG_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)

def get_market_file(args):
    '''Get path to market data file based on the run mode'''
    if args.debug: csv_filenm =SP500_MARKET_DEBUG
    elif args.test: csv_filenm = SP500_MARKET_TEST
    else: csv_filenm = SP500_MARKET_FILE
    print("opening", csv_filenm)
    return csv_filenm

def get_sentiment_file(args):
    """Get path to sentiment data CSV"""
    data_dir = get_dir(args, 'data')
    
    if args.test:
        return data_dir / "sentiment_test.csv"
    else:
        return data_dir / "sentiment.csv"
    

def get_fundamental_file(args):
    '''Get path to fundamental data file based on the run mode'''
    if args.debug: csv_filenm =SP500_FUNDA_DEBUG
    elif args.test: csv_filenm = SP500_FUNDA_TEST
    else: csv_filenm = SP500_FUNDA_FILE
    print("opening", csv_filenm)
    return csv_filenm



def get_dir(args, dir_type):
    '''Function to determine which folder/subfolder to choose or create'''
    if args.debug:
        file_dir = DEBUG_DIR
    elif 'data' in dir_type.lower():
        file_dir = DATA_DIR
    elif 'model' in dir_type.lower():
        file_dir = MODELS_DIR
    elif 'figure' in dir_type.lower():
        file_dir = FIGURE_DIR
    else:
        print("file_dir not generated, please check", args, dir_type)
        exit()

    if hasattr(args, 'only_market') and args.only_market:
        file_dir = file_dir / 'only_market'
        file_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(args, 'use_enhanced') and args.use_enhanced:
        file_dir = file_dir / 'enhanced'
        file_dir.mkdir(parents=True, exist_ok=True)
    return file_dir

def get_prediction_file(args):
    '''Get path to predition output file based on the run mode'''
    #file_dir = DEBUG_DIR if args.debug else DATA_DIR
    file_dir = get_dir(args, 'data')

    enhanced = '_enhanced' if getattr(args, 'use_enhanced', False) else ''
    market_suffix = '_only_market' if  getattr(args, 'only_market', False) else ''
    if args.debug: file_name = f'sp500_oof_predictions{market_suffix}{enhanced}_debug.csv'
    elif args.test: file_name = f'sp500_oof_predictions{market_suffix}{enhanced}_test.csv'
    else: file_name = f'sp500_oof_predictions{market_suffix}{enhanced}.csv'
    print("Chosen prediction file:", file_dir / file_name)
    return file_dir / file_name

def get_model_file(args, model_name):
    '''Define path to model file based on the run mode'''

    file_dir = get_dir(args, 'model')

    if args.debug: file_name = f'{model_name}_debug.pkl'
    elif args.test: file_name = f'{model_name}_test.pkl'
    else: file_name = f'{model_name}.pkl'
    print("Chosen model file:", file_dir / file_name)
    return file_dir / file_name

def get_features_file(args):
    '''Define path to file with features based on the run mode'''
    file_dir = get_dir(args, 'model')
    market_suffix = '_only_market' if args.only_market else ''
    if args.debug: file_name = f'feature_names{market_suffix}_debug.txt'
    elif args.test: file_name = f'feature_names{market_suffix}_test.txt'
    else: file_name = f'feature_names{market_suffix}.txt'
    print("Chosen feature file:", file_dir / file_name)
    return file_dir / file_name