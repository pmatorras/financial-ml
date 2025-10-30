"""
Data collection from external sources.

Collectors:
- collect_market_data(): Download stock prices from yfinance
- collect_fundamentals(): Download company fundamentals from SEC EDGAR

Usage:
    from financial_ml.data.collectors import collect_market_data, collect_fundamentals
    
    collect_market_data(args)
    collect_fundamentals(args)
"""

from financial_ml.data.collectors.market_data import collect_market_data
from financial_ml.data.collectors.fundamental_data import collect_fundamentals_data
from financial_ml.data.collectors.sentiment_data import collect_sentiment_data

__all__ = [
    'collect_market_data',
    'collect_fundamentals_data',
    'collect_sentiment_data'
]
