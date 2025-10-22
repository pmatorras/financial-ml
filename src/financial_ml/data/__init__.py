"""
Data loading, feature engineering, and collection.

Main modules:
- loaders: Load processed CSV data
- features: Feature engineering pipeline
- collectors: Download raw data from external sources
"""

# You can optionally expose key loaders here
from financial_ml.data.loaders import load_market, load_fundamentals
from financial_ml.data.features import prepare_features, create_labels

__all__ = [
    'load_market',
    'load_fundamentals',
    'prepare_features',
    'create_labels',
]