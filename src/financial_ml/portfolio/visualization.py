"""
Portfolio visualization functions.
"""

import matplotlib.pyplot as plt
from financial_ml.utils.config import FIGURE_DIR
from financial_ml.models.definitions import get_model_name

def draw_cumulative_drawdown(portfolio_returns,spy, drawdown, max_drawdown, model):
    """
    Create 2-panel chart: cumulative returns and drawdown.
    
    Args:
        portfolio_returns: DataFrame with 'date', 'cum_return', 'spy_cum_return'
        model_key: Model identifier (e.g., 'logreg_l2')
    """
    # Visualizations
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    model_name = get_model_name(model_key=model)
    model_name_short = get_model_name(model_key=model, short_name=True)

    # Chart 1: Cumulative Returns
    axes[0].plot(portfolio_returns['date'], portfolio_returns['cum_return'], 
                label=f'ML Strategy ({model_name_short})', linewidth=2, color='blue')
    if spy is not None:
        axes[0].plot(portfolio_returns['date'], portfolio_returns['spy_cum_return'], 
                    label='SPY (Buy & Hold)', linewidth=2, color='gray', linestyle='--')
    axes[0].set_title(f'Cumulative Returns: ML Strategy ({model_name}) vs SPY', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return (1 = start)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Chart 2: Drawdown Over Time
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
    figname = FIGURE_DIR / f"portfolio_backtest_{model}.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    #plt.show()

    print(f"\nChart saved to {figname}")
