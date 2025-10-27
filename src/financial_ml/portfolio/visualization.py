"""
Portfolio visualization functions.
"""

import matplotlib.pyplot as plt
import pandas as pd
from financial_ml.utils.config import FIGURE_DIR
from financial_ml.models import get_model_name



def draw_cumulative_drawdown_all(portfolio_returns, spy, equal_weight_returns, random_returns,
                            drawdown, max_drawdown, model, portfolio_type, per_top):
    """
    Create 2-panel chart with honest performance attribution.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    model_name = get_model_name(model_key=model)
    model_name_short = get_model_name(model_key=model, short_name=True)

    # Chart 1: Cumulative Returns with Multiple Benchmarks
    axes[0].plot(portfolio_returns['date'], portfolio_returns['cum_return'], 
                label=f'Your Model (Top {per_top}%)', linewidth=2.5, color='#2E7D32', zorder=4)
    
    # Handle random_returns - might be None or DataFrame
    if random_returns is not None:
        if isinstance(random_returns, pd.DataFrame):
            # It's already a DataFrame with 'date' and 'cum_return'
            axes[0].plot(random_returns['date'], random_returns['cum_return'], 
                        label=f'Random Selection (Top {per_top}%)', linewidth=2, 
                        color='#FFA726', linestyle='--', alpha=0.8, zorder=3)
        else:
            # It's a Series - compute cumulative on the fly
            cum_random = (1 + random_returns).cumprod()
            axes[0].plot(portfolio_returns['date'], cum_random, 
                        label=f'Random Selection (Top {per_top}%)', linewidth=2, 
                        color='#FFA726', linestyle='--', alpha=0.8, zorder=3)
    
    # Handle equal_weight_returns - might be None or DataFrame
    if equal_weight_returns is not None:
        if isinstance(equal_weight_returns, pd.DataFrame):
            axes[0].plot(equal_weight_returns['date'], equal_weight_returns['cum_return'], 
                        label='Equal-Weight Benchmark (100%)', linewidth=2, 
                        color='#1976D2', linestyle='-.', alpha=0.8, zorder=2)
        else:
            cum_equal = (1 + equal_weight_returns).cumprod()
            axes[0].plot(portfolio_returns['date'], cum_equal, 
                        label='Equal-Weight Benchmark (100%)', linewidth=2, 
                        color='#1976D2', linestyle='-.', alpha=0.8, zorder=2)
    
    # SPY (your original code)
    if spy is not None:
        axes[0].plot(portfolio_returns['date'], portfolio_returns['spy_cum_return'], 
                    label='SPY (Cap-Weighted)', linewidth=2, color='#757575', 
                    linestyle=':', alpha=0.7, zorder=1)
    
    axes[0].set_title(f'Performance Attribution: {model_name} Strategy', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].grid(alpha=0.3)

    # Chart 2: Drawdown (unchanged)
    axes[1].fill_between(portfolio_returns['date'], drawdown * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
    axes[1].axhline(max_drawdown * 100, color='darkred', linestyle='--', 
                    label=f'Max Drawdown: {max_drawdown:.1%}')
    axes[1].set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    figname = FIGURE_DIR / f"portfolio_backtest_{model}_{portfolio_type}_top{per_top}.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    
    print(f"\nChart saved to {figname}")

