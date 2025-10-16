from pathlib import Path
from typing import Optional
from financial_ml.utils.config import (
    DATA_DIR, FIGURE_DIR, LOGS_DIR, TEST_DIR, DEBUG_DIR,
    SP500_MARKET_FILE, SP500_MARKET_TEST, SP500_MARKET_DEBUG,
    SP500_FUNDA_FILE, SP500_FUNDA_TEST, SP500_FUNDA_DEBUG,
    SP500_PRED_FILE, SP500_PRED_TEST, SP500_PRED_DEBUG
)

def createFolders():
    """Create all required project directories if they don't exist."""
    for p in (DATA_DIR,FIGURE_DIR, LOGS_DIR, TEST_DIR, DEBUG_DIR):
        p.mkdir(parents=True, exist_ok=True)

def get_market_file(args):
    '''Get path to market data file based on the run mode'''
    if args.debug: csv_filenm =SP500_MARKET_DEBUG
    elif args.test: csv_filenm = SP500_MARKET_TEST
    else: csv_filenm = SP500_MARKET_FILE
    print("opening", csv_filenm)
    return csv_filenm

def get_fundamental_file(args):
    '''Get path to fundamental data file based on the run mode'''
    if args.debug: csv_filenm =SP500_FUNDA_DEBUG
    elif args.test: csv_filenm = SP500_FUNDA_TEST
    else: csv_filenm = SP500_FUNDA_FILE
    print("opening", csv_filenm)
    return csv_filenm

def get_prediction_file(args):
    '''Get path to predition output file based on the run mode'''
    if args.debug: csv_filenm =SP500_PRED_DEBUG
    elif args.test: csv_filenm = SP500_PRED_TEST
    else: csv_filenm = SP500_PRED_FILE
    print("opening", csv_filenm)
    return csv_filenm