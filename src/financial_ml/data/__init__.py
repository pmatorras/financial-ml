"""
Data loading, feature engineering, and collection.

Main modules:
- loaders: Load processed CSV data
- features: Feature engineering pipeline
- collectors: Download raw data from external sources
"""

# You can optionally expose key loaders here
from financial_ml.data.validation import require_non_empty
from financial_ml.data.loaders import load_market, load_fundamentals
from financial_ml.data.features import (
    calculate_market_features,
    compute_fundamental_ratios,
    create_binary_labels,
    to_monthly_ffill,
    widen_by_canonical
)

__all__ = [
    'require_non_empty',
    'load_market',
    'load_fundamentals',
    'calculate_market_features',
    'compute_fundamental_ratios', 
    'create_binary_labels',
    'to_monthly_ffill',
    'widen_by_canonical'
]