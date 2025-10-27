# src/financial_ml/portfolio/__init__.py
"""
Portfolio module for backtesting and performance analysis.

Provides:
- Portfolio construction from predictions
- Performance calculation and metrics
- Diagnostic analysis tools
- Visualization functions
"""

# Construction
from financial_ml.portfolio.construction import (
    construct_portfolio,
    smooth_predictions
)

# Performance
from financial_ml.portfolio.performance import (
    aggregate_portfolio_return,
    include_benchmark_return
)

# Visualization
from financial_ml.portfolio.visualization import (
    plot_cumulative_drawdown_all,
    plot_sector_concentration_over_time
)

# Diagnostics - import commonly used ones explicitly
from financial_ml.portfolio.diagnostics import (
    pre_filter_diagnostics,
    print_model_agreement,
    model_agreement_correlations,
    compare_model_performance_by_period,
    analyze_prediction_stability,
    analyze_turnover,
    analyze_beta_exposure,
    analyze_sector_concentration,
    compare_drawdowns_to_spy,
    test_sharpe_significance
)

# Main backtest function
from financial_ml.portfolio.backtest import run_backtest


# Define what gets imported with "from financial_ml.portfolio import *"
__all__ = [
    # Construction
    'construct_portfolio',
    'smooth_predictions',
    
    # Performance
    'aggregate_portfolio_return',
    'include_benchmark_return',
    
    # Visualization
    'draw_cumulative_drawdown',
    
    # Diagnostics
    'pre_filter_diagnostics',
    'print_model_agreement',
    'model_agreement_correlations',
    'compare_model_performance_by_period',
    'analyze_prediction_stability',
    'analyze_turnover',
    'analyze_beta_exposure',
    'analyze_sector_concentration',
    'compare_drawdowns_to_spy',
    'test_sharpe_significance',
    
    # Main
    'run_backtest',
]
